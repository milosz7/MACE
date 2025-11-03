import copy
import torch
import numpy as np
from tqdm import tqdm
import gc
from typing import Any
import torch.nn.functional as F


def find_matching_indices(old, new):
    # Find the starting common sequence
    start_common = 0
    for i, j in zip(old, new):
        if i == j:
            start_common += 1
        else:
            break

    # Find the ending common sequence
    end_common_old = len(old) - 1
    end_common_new = len(new) - 1
    while end_common_old >= start_common and end_common_new >= start_common:
        if old[end_common_old] == new[end_common_new]:
            end_common_old -= 1
            end_common_new -= 1
        else:
            break

    return list(range(start_common)) + list(range(end_common_old + 1, len(old))), \
           list(range(start_common)) + list(range(end_common_new + 1, len(new)))


def get_ca_layers(unet, with_to_k=True):
    sub_nets = unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ## get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    return projection_matrices, ca_layers, og_matrices


def prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, test_set,
                tokenizer, with_to_k=True, all_words=False, prepare_k_v_for_lora=False, input_projections=None):
    with torch.no_grad():
        all_contexts, all_valuess = [], []

        for curr_item in test_set:
            gc.collect()
            torch.cuda.empty_cache()

            #### restart LDM parameters
            num_ca_clip_layers = len(ca_layers)
            for idx_, l in enumerate(ca_layers):
                l.to_v = copy.deepcopy(og_matrices[idx_])
                projection_matrices[idx_] = l.to_v
                if with_to_k:
                    l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
                    projection_matrices[num_ca_clip_layers + idx_] = l.to_k

            old_embs, new_embs = [], []
            extended_old_indices, extended_new_indices = [], []

            #### indetify corresponding destinations for each token in old_emb
            # Bulk tokenization
            texts_old = [item[0] for item in curr_item["old"]]
            texts_new = [item[0] for item in curr_item["new"]]
            texts_combined = texts_old + texts_new

            tokenized_inputs = tokenizer(
                texts_combined,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )

            # Text embeddings
            text_embeddings = text_encoder(tokenized_inputs.input_ids.to(text_encoder.device))[0]
            # embedding dim
            try:
                emb_dim = text_embeddings.shape[2]
            except Exception:
                emb_dim = None
            # local adapters cache: map target_in -> projection module
            local_adapters = {}
            old_embs.extend(text_embeddings[:len(texts_old)])
            new_embs.extend(text_embeddings[len(texts_old):])

            # Find matching indices
            for old_text, new_text in zip(texts_old, texts_new):
                tokens_a = tokenizer(old_text).input_ids
                tokens_b = tokenizer(new_text).input_ids

                old_indices, new_indices = find_matching_indices(tokens_a, tokens_b)

                if old_indices[-1] >= new_indices[-1]:
                    extended_old_indices.append(old_indices + list(range(old_indices[-1] + 1, 77)))
                    extended_new_indices.append(
                        new_indices + list(range(new_indices[-1] + 1, 77 - (old_indices[-1] - new_indices[-1]))))
                else:
                    extended_new_indices.append(new_indices + list(range(new_indices[-1] + 1, 77)))
                    extended_old_indices.append(
                        old_indices + list(range(old_indices[-1] + 1, 77 - (new_indices[-1] - old_indices[-1]))))

            #### prepare batch: for each pair of setences, old context and new values
            contexts, valuess = [], []
            if not all_words:
                for idx, (old_emb, new_emb) in enumerate(zip(old_embs, new_embs)):
                    context = old_emb[extended_old_indices[idx]].detach()
                    values = []
                    for layer_idx, layer in enumerate(projection_matrices):
                        inp = new_emb[extended_new_indices[idx]]
                        # If an input projection is provided, project the embedding and store the projected input.
                        if input_projections is not None:
                            proj = input_projections[layer_idx]
                            proj = proj.to(layer.weight.device)
                            inp_proj = proj(inp.to(proj.weight.device)).detach()
                            values.append(inp_proj)
                        else:
                            # create/reuse a local adapter projecting emb_dim -> layer.in_features
                            target_in = layer.weight.shape[1]
                            if emb_dim is None:
                                # fallback: assume embedding dim equals target_in and apply layer directly
                                inp_dev = inp.to(layer.weight.device)
                                values.append(layer(inp_dev).detach())
                            else:
                                if target_in not in local_adapters:
                                    proj = torch.nn.Linear(emb_dim, target_in, bias=True)
                                    with torch.no_grad():
                                        proj.weight.zero_()
                                        proj.bias.zero_()
                                        overlap = min(emb_dim, target_in)
                                        if overlap > 0:
                                            proj.weight[:overlap, :overlap] = torch.eye(overlap)
                                    proj.to(layer.weight.device)
                                    proj.requires_grad_(False)
                                    local_adapters[target_in] = proj
                                proj = local_adapters[target_in]
                                inp_proj = proj(inp.to(proj.weight.device)).detach()
                                values.append(inp_proj)
                    contexts.append(context)
                    valuess.append(values)

                all_contexts.append(contexts)
                all_valuess.append(valuess)
            else:
                if prepare_k_v_for_lora:
                    # prepare for lora, then no need to use new_emb
                    for idx, old_emb in enumerate(old_embs):
                        context = old_emb.detach()
                        values = []
                        for layer_idx, layer in enumerate(projection_matrices):
                            inp = old_emb
                            if input_projections is not None:
                                proj = input_projections[layer_idx]
                                proj = proj.to(layer.weight.device)
                                inp_proj = proj(inp.to(proj.weight.device)).detach()
                                values.append(inp_proj)
                            else:
                                target_in = layer.weight.shape[1]
                                if emb_dim is None:
                                    inp_dev = inp.to(layer.weight.device)
                                    values.append(layer(inp_dev).detach())
                                else:
                                    if target_in not in local_adapters:
                                        proj = torch.nn.Linear(emb_dim, target_in, bias=True)
                                        with torch.no_grad():
                                            proj.weight.zero_()
                                            proj.bias.zero_()
                                            overlap = min(emb_dim, target_in)
                                            if overlap > 0:
                                                proj.weight[:overlap, :overlap] = torch.eye(overlap)
                                        proj.to(layer.weight.device)
                                        proj.requires_grad_(False)
                                        local_adapters[target_in] = proj
                                    proj = local_adapters[target_in]
                                    inp_proj = proj(inp.to(proj.weight.device)).detach()
                                    values.append(inp_proj)
                        contexts.append(context)
                        valuess.append(values)
                else:
                    # need to use new_emb
                    for idx, (old_emb, new_emb) in enumerate(zip(old_embs, new_embs)):
                        context = old_emb.detach()
                        values = []
                        for layer_idx, layer in enumerate(projection_matrices):
                            inp = new_emb
                            if input_projections is not None:
                                proj = input_projections[layer_idx]
                                proj = proj.to(layer.weight.device)
                                inp_proj = proj(inp.to(proj.weight.device)).detach()
                                values.append(inp_proj)
                            else:
                                target_in = layer.weight.shape[1]
                                if emb_dim is None:
                                    inp_dev = inp.to(layer.weight.device)
                                    values.append(layer(inp_dev).detach())
                                else:
                                    if target_in not in local_adapters:
                                        proj = torch.nn.Linear(emb_dim, target_in, bias=True)
                                        with torch.no_grad():
                                            proj.weight.zero_()
                                            proj.bias.zero_()
                                            overlap = min(emb_dim, target_in)
                                            if overlap > 0:
                                                proj.weight[:overlap, :overlap] = torch.eye(overlap)
                                        proj.to(layer.weight.device)
                                        proj.requires_grad_(False)
                                        local_adapters[target_in] = proj
                                    proj = local_adapters[target_in]
                                    inp_proj = proj(inp.to(proj.weight.device)).detach()
                                    values.append(inp_proj)
                        contexts.append(context)
                        valuess.append(values)

                all_contexts.append(contexts)
                all_valuess.append(valuess)

        return all_contexts, all_valuess


def closed_form_refinement(projection_matrices, all_contexts=None, all_valuess=None, lamb=0.5,
                           preserve_scale=1, cache_dict=None, cache_dict_path=None, cache_mode=False,
                           input_projections=None):
    with torch.no_grad():
        if cache_dict_path is not None:
            cache_dict = torch.load(cache_dict_path, map_location=projection_matrices[0].weight.device)

        for layer_num in tqdm(range(len(projection_matrices))):
            gc.collect()
            torch.cuda.empty_cache()

            mat1 = lamb * projection_matrices[layer_num].weight
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1],
                                    device=projection_matrices[layer_num].weight.device)

            total_for_mat1 = torch.zeros_like(projection_matrices[layer_num].weight)
            total_for_mat2 = torch.zeros_like(mat2)

            if all_contexts is not None and all_valuess is not None:
                # Aggregate over all provided (contexts, valuess) pairs.
                # For each pair, contexts is a list of context tensors (T, emb_dim) and
                # valuess is a list where valuess[i][layer_num] is a tensor (T, out_feat).
                # We'll compute per-sample: for_mat1_s = V_s.T @ C_s  -> (out_feat, in_feat)
                #                         for_mat2_s = C_s.T @ C_s  -> (in_feat, in_feat)
                for contexts, valuess in zip(all_contexts, all_valuess):
                    for c_idx, c in enumerate(contexts):
                        # c: (T, emb_dim)
                        val_item = valuess[c_idx][layer_num]
                        # Move tensors to layer device
                        target_dev = projection_matrices[layer_num].weight.device
                        val_item = val_item.to(target_dev)

                        layer = projection_matrices[layer_num]
                        out_feat = layer.weight.shape[0]
                        in_feat = layer.weight.shape[1]

                        # Normalize val_item dims
                        if val_item.dim() == 1:
                            val_item = val_item.unsqueeze(0)

                        # Determine v: desired shape (T, out_feat)
                        if val_item.shape[1] == in_feat:
                            bias = layer.bias if hasattr(layer, 'bias') else None
                            v = F.linear(val_item, layer.weight, bias)
                        elif val_item.shape[1] == out_feat:
                            v = val_item
                        else:
                            # adapt val_item to in_feat by truncating/padding
                            cur = val_item.shape[1]
                            if cur < in_feat:
                                padded = val_item.new_zeros((val_item.shape[0], in_feat))
                                padded[:, :cur] = val_item
                                val_adj = padded
                            else:
                                val_adj = val_item[:, :in_feat]
                            bias = layer.bias if hasattr(layer, 'bias') else None
                            v = F.linear(val_adj, layer.weight, bias)

                        # Now compute c_proj: contexts must be projected to in_feat when input_projections provided,
                        # otherwise contexts are already in the expected in_feat.
                        if input_projections is not None:
                            proj = input_projections[layer_num]
                            proj = proj.to(target_dev)
                            c_proj = proj(c.to(proj.weight.device)).to(target_dev)
                        else:
                            c_proj = c.to(target_dev)

                        if c_proj.dim() == 1:
                            c_proj = c_proj.unsqueeze(0)

                        # Ensure c_proj has second-dim == in_feat by truncating/padding if needed
                        cur_c = c_proj.shape[1]
                        if cur_c != in_feat:
                            if cur_c < in_feat:
                                padded = c_proj.new_zeros((c_proj.shape[0], in_feat))
                                padded[:, :cur_c] = c_proj
                                c_proj = padded
                            else:
                                c_proj = c_proj[:, :in_feat]

                        # Ensure v has shape (T, out_feat)
                        if v.dim() == 1:
                            v = v.unsqueeze(0)
                        if v.shape[1] != out_feat:
                            # adjust v by truncation/padding on second dim
                            cur_v = v.shape[1]
                            if cur_v < out_feat:
                                padded = v.new_zeros((v.shape[0], out_feat))
                                padded[:, :cur_v] = v
                                v = padded
                            else:
                                v = v[:, :out_feat]

                        # Now compute matrices
                        # Align time dimensions (T) between v and c_proj so matmul is possible.
                        T_v = v.shape[0]
                        T_c = c_proj.shape[0]
                        if T_v != T_c:
                            # If v has single timestep, expand it to match c_proj
                            if T_v == 1:
                                v = v.expand(T_c, -1).contiguous()
                                T_v = T_c
                            # If c_proj has single timestep, expand it to match v
                            elif T_c == 1:
                                c_proj = c_proj.expand(T_v, -1).contiguous()
                                T_c = T_v
                            else:
                                # Neither is 1: truncate the longer one to match the shorter
                                if T_v > T_c:
                                    v = v[:T_c, :]
                                    T_v = T_c
                                else:
                                    # pad v with zeros to match T_c
                                    pad = v.new_zeros((T_c - T_v, v.shape[1]))
                                    v = torch.cat([v, pad], dim=0)
                                    T_v = T_c

                        # Recompute now that T_v == T_c
                        for_mat1 = v.transpose(0, 1) @ c_proj
                        for_mat2 = c_proj.transpose(0, 1) @ c_proj

                        # Make sure for_mat1 and for_mat2 match the projection_matrices shapes before accumulation
                        if for_mat1.shape != total_for_mat1.shape:
                            tmp = torch.zeros_like(total_for_mat1)
                            r = min(tmp.shape[0], for_mat1.shape[0])
                            c_ = min(tmp.shape[1], for_mat1.shape[1])
                            tmp[:r, :c_] = for_mat1[:r, :c_]
                            for_mat1 = tmp
                        if for_mat2.shape != total_for_mat2.shape:
                            tmp2 = torch.zeros_like(total_for_mat2)
                            r2 = min(tmp2.shape[0], for_mat2.shape[0])
                            c2 = min(tmp2.shape[1], for_mat2.shape[1])
                            tmp2[:r2, :c2] = for_mat2[:r2, :c2]
                            for_mat2 = tmp2

                        total_for_mat1 += for_mat1
                        total_for_mat2 += for_mat2

                del for_mat1, for_mat2

            if cache_mode:
                # cache the results
                if cache_dict[f'{layer_num}_for_mat1'] is None:
                    cache_dict[f'{layer_num}_for_mat1'] = total_for_mat1
                    cache_dict[f'{layer_num}_for_mat2'] = total_for_mat2
                else:
                    cache_dict[f'{layer_num}_for_mat1'] += total_for_mat1
                    cache_dict[f'{layer_num}_for_mat2'] += total_for_mat2
            else:
                # CFR calculation
                if cache_dict_path is not None or cache_dict is not None:
                    total_for_mat1 += preserve_scale * cache_dict[f'{layer_num}_for_mat1']
                    total_for_mat2 += preserve_scale * cache_dict[f'{layer_num}_for_mat2']

                total_for_mat1 += mat1
                total_for_mat2 += mat2

                projection_matrices[layer_num].weight.data = total_for_mat1 @ torch.inverse(total_for_mat2)

            del total_for_mat1, total_for_mat2


def importance_sampling_fn(t, temperature=0.05):
    """Importance Sampling Function f(t)"""
    return 1 / (1 + np.exp(-temperature * (t - 200))) - 1 / (1 + np.exp(-temperature * (t - 400)))


class AttnController:
    def __init__(self) -> None:
        self.attn_probs = []
        self.logs = []

    def __call__(self, attn_prob, m_name, preserve_prior, latent_num) -> Any:
        bs, _ = self.concept_positions.shape

        if preserve_prior:
            attn_prob = attn_prob[:attn_prob.shape[0] // latent_num]

        if self.use_gsam_mask:
            d = int(attn_prob.shape[1] ** 0.5)
            resized_mask = F.interpolate(self.mask, size=(d, d), mode='nearest')

            # # save mask
            # img_array = (resized_mask > 0.5).to(torch.uint8) * 255
            # from PIL import Image
            # img = Image.fromarray(img_array[0][0].cpu().numpy())
            # img.save('./sam_outputs/bool_image.png')

            resized_mask = (resized_mask > 0.5).view(-1)
            attn_prob = attn_prob[:, resized_mask, :]
            target_attns = attn_prob[:, :, self.concept_positions[0]]
        else:
            head_num = attn_prob.shape[0] // bs
            target_attns = attn_prob.masked_select(self.concept_positions[:, None, :].repeat(head_num, 1, 1)).reshape(-1, self.concept_positions[0].sum())

        self.attn_probs.append(target_attns)
        self.logs.append(m_name)

    def set_concept_positions(self, concept_positions, mask=None, use_gsam_mask=False):
        self.concept_positions = concept_positions
        self.mask = mask
        self.use_gsam_mask = use_gsam_mask

    def loss(self):
        return sum(torch.norm(item) for item in self.attn_probs)

    def zero_attn_probs(self):
        self.attn_probs = []
        self.logs = []
        self.concept_positions = None


def prompt_augmentation(content, augment=True, sampled_indices=None, concept_type='object'):
    if augment:
        # some sample prompts provided
        if concept_type == 'object':
            prompts = [
                # object augmentation
                ("{} in a photo".format(content), content),
                ("{} in a snapshot".format(content), content),
                ("A snapshot of {}".format(content), content),
                ("A photograph showcasing {}".format(content), content),
                ("An illustration of {}".format(content), content),
                ("A digital rendering of {}".format(content), content),
                ("A visual representation of {}".format(content), content),
                ("A graphic of {}".format(content), content),
                ("A shot of {}".format(content), content),
                ("A photo of {}".format(content), content),
                ("A black and white image of {}".format(content), content),
                ("A depiction in portrait form of {}".format(content), content),
                ("A scene depicting {} during a public gathering".format(content), content),
                ("{} captured in an image".format(content), content),
                ("A depiction created with oil paints capturing {}".format(content), content),
                ("An image of {}".format(content), content),
                ("A drawing capturing the essence of {}".format(content), content),
                ("An official photograph featuring {}".format(content), content),
                ("A detailed sketch of {}".format(content), content),
                ("{} during sunset/sunrise".format(content), content),
                ("{} in a detailed portrait".format(content), content),
                ("An official photo of {}".format(content), content),
                ("Historic photo of {}".format(content), content),
                ("Detailed portrait of {}".format(content), content),
                ("A painting of {}".format(content), content),
                ("HD picture of {}".format(content), content),
                ("Magazine cover capturing {}".format(content), content),
                ("Painting-like image of {}".format(content), content),
                ("Hand-drawn art of {}".format(content), content),
                ("An oil portrait of {}".format(content), content),
                ("{} in a sketch painting".format(content), content),
            ]

        elif concept_type == 'style':
            # art augmentation
            prompts = [
                ("An artwork by {}".format(content), content),
                ("Art piece by {}".format(content), content),
                ("A recent creation by {}".format(content), content),
                ("{}'s renowned art".format(content), content),
                ("Latest masterpiece by {}".format(content), content),
                ("A stunning image by {}".format(content), content),
                ("An art in {}'s style".format(content), content),
                ("Exhibition artwork of {}".format(content), content),
                ("Art display by {}".format(content), content),
                ("a beautiful painting by {}".format(content), content),
                ("An image inspired by {}'s style".format(content), content),
                ("A sketch by {}".format(content), content),
                ("Art piece representing {}".format(content), content),
                ("A drawing by {}".format(content), content),
                ("Artistry showcasing {}".format(content), content),
                ("An illustration by {}".format(content), content),
                ("A digital art by {}".format(content), content),
                ("A visual art by {}".format(content), content),
                ("A reproduction inspired by {}'s colorful, expressive style".format(content), content),
                ("Famous painting of {}".format(content), content),
                ("A famous art by {}".format(content), content),
                ("Artistic style of {}".format(content), content),
                ("{}'s famous piece".format(content), content),
                ("Abstract work of {}".format(content), content),
                ("{}'s famous drawing".format(content), content),
                ("Art from {}'s early period".format(content), content),
                ("A portrait by {}".format(content), content),
                ("An imitation reflecting the style of {}".format(content), content),
                ("An painting from {}'s collection".format(content), content),
                ("Vibrant reproduction of artwork by {}".format(content), content),
                ("Artistic image influenced by {}".format(content), content),
            ]
        else:
            raise ValueError("unknown concept type.")
    else:
        prompts = [
            ("A photo of {}".format(content), content),
        ]

    if sampled_indices is not None:
        sampled_prompts = [prompts[i] for i in sampled_indices if i < len(prompts)]
    else:
        sampled_prompts = prompts

    return sampled_prompts
