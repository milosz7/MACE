import os, gc
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from omegaconf import OmegaConf
import argparse
import torch.nn as nn

def save_images(images, output_folder, prompt):
    n = 0
    filename = f"{output_folder}/{prompt.replace(' ', '-')}_{n}.jpg"
    while os.path.exists(filename):
        n += 1
        filename = f"{output_folder}/{prompt.replace(' ', '-')}_{n}.jpg"
    for i, im in enumerate(images, start=n):
        filename = f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg"
        print(f"SAVING IMAGE {filename}")
        im.save(filename)


def main(args):

    model_id = args.pretrained_model_name_or_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(args.device)
    # Ensure UNet won't expect additional text embeds by default (prevents text-embed shape errors)
    try:
        pipe.unet.config.addition_embed_type = "none"
    except Exception:
        pass
    # Build a small inference-time projection to avoid matmul shape mismatches
    # Map text encoder embedding dim -> UNet cross-attention dim when they differ.
    try:
        # infer embedding dim from text encoder config or embeddings
        try:
            emb_dim = int(pipe.text_encoder.config.hidden_size)
        except Exception:
            try:
                emb_dim = pipe.text_encoder.get_input_embeddings().weight.shape[1]
            except Exception:
                emb_dim = 768

        # try to read cross attention dim from unet config
        try:
            cross_attn_dim = pipe.unet.config.cross_attention_dim
        except Exception:
            cross_attn_dim = None

        # fallback: inspect first attn processor's to_k weight in_features
        if cross_attn_dim is None:
            try:
                for name, proc in pipe.unet.attn_processors.items():
                    if hasattr(proc, "to_k") and hasattr(proc.to_k, "weight"):
                        cross_attn_dim = proc.to_k.weight.shape[1]
                        break
            except Exception:
                cross_attn_dim = None

        if cross_attn_dim is None:
            cross_attn_dim = emb_dim

        # Prepare a placeholder for a simple global projection (may stay None).
        global_text_proj = None

        # Create a global projection adapter so the text encoder's outputs match
        # the UNet expected cross-attention dim. We will also monkeypatch
        # `pipe.text_encoder.forward` to apply this projection to any returned
        # encoder hidden states, which is the simplest robust fix.
        try:
            if emb_dim != cross_attn_dim:
                global_text_proj = nn.Linear(emb_dim, cross_attn_dim, bias=True)
                with torch.no_grad():
                    global_text_proj.weight.zero_()
                    global_text_proj.bias.zero_()
                    overlap = min(emb_dim, cross_attn_dim)
                    if overlap > 0:
                        for d in range(overlap):
                            global_text_proj.weight[d, d] = 1.0
                # Move proj to the same device as the UNet
                try:
                    proj_device = next(pipe.unet.parameters()).device
                except Exception:
                    proj_device = args.device
                global_text_proj.to(proj_device)
                global_text_proj.requires_grad_(False)

                # Monkeypatch text_encoder.forward to project last_hidden_state / first value
                try:
                    orig_te_forward = pipe.text_encoder.forward

                    def _patched_text_encoder_forward(*f_args, **f_kwargs):
                        out = orig_te_forward(*f_args, **f_kwargs)
                        try:
                            # If output is tuple/list, project first element
                            if isinstance(out, (list, tuple)) and len(out) > 0:
                                first = out[0]
                                if hasattr(first, 'shape') and first.shape[-1] == emb_dim:
                                    first = first.to(next(global_text_proj.parameters()).device)
                                    first = global_text_proj(first)
                                    out = (first,) + tuple(out[1:])
                            # If output has attribute last_hidden_state (transformers BaseModelOutput)
                            elif hasattr(out, 'last_hidden_state'):
                                lhs = getattr(out, 'last_hidden_state')
                                if lhs is not None and hasattr(lhs, 'shape') and lhs.shape[-1] == emb_dim:
                                    lhs = lhs.to(next(global_text_proj.parameters()).device)
                                    lhs = global_text_proj(lhs)
                                    try:
                                        setattr(out, 'last_hidden_state', lhs)
                                    except Exception:
                                        # fallback to creating a new simple object/dict
                                        try:
                                            out.last_hidden_state = lhs
                                        except Exception:
                                            pass
                            # If it's a dict-like output
                            elif isinstance(out, dict) and 'last_hidden_state' in out:
                                lhs = out['last_hidden_state']
                                if lhs is not None and hasattr(lhs, 'shape') and lhs.shape[-1] == emb_dim:
                                    lhs = lhs.to(next(global_text_proj.parameters()).device)
                                    lhs = global_text_proj(lhs)
                                    out['last_hidden_state'] = lhs
                        except Exception:
                            pass
                        return out

                    try:
                        pipe.text_encoder.forward = _patched_text_encoder_forward
                        print('Patched pipe.text_encoder.forward to project text embeddings to UNet cross-attn dim')
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            global_text_proj = None

        # Helper wrapper used to project encoder hidden states before calling the
        # original to_k callable. Defined early so it can be referenced by later
        # monkeypatch/wrapping code.
    #     class _ProjectedToK(torch.nn.Module):
    #         def __init__(self, orig_to_k, proj_module, emb_dim, expected_in, name=None):
    #             super().__init__()
    #             if isinstance(orig_to_k, torch.nn.Module):
    #                 self.orig = orig_to_k
    #             else:
    #                 class _CallableWrapper(torch.nn.Module):
    #                     def __init__(self, fn):
    #                         super().__init__()
    #                         self.fn = fn
    #
    #                     def forward(self, *a, **k):
    #                         return self.fn(*a, **k)
    #
    #                 self.orig = _CallableWrapper(orig_to_k)
    #
    #             self.proj = proj_module
    #             self.emb_dim = emb_dim
    #             self.expected_in = expected_in
    #             self.name = name or "<to_k>"
    #
    #         def forward(self, *f_args, **f_kwargs):
    #             if len(f_args) > 0:
    #                 x = f_args[0]
    #                 rest = f_args[1:]
    #             elif "hidden_states" in f_kwargs:
    #                 x = f_kwargs["hidden_states"]
    #                 f_kwargs.pop("hidden_states")
    #                 rest = ()
    #             else:
    #                 return self.orig(*f_args, **f_kwargs)
    #
    #             try:
    #                 debug = os.environ.get("MACE_DEBUG_ATTN", "0") in ("1", "true", "True")
    #                 in_shape = None
    #                 if x is not None:
    #                     try:
    #                         in_shape = tuple(x.shape)
    #                     except Exception:
    #                         in_shape = str(type(x))
    #
    #                 if x is not None and x.shape[-1] == self.emb_dim and self.proj is not None:
    #                     x = x.to(next(self.proj.parameters()).device)
    #                     x = self.proj(x)
    #
    #                 if debug:
    #                     try:
    #                         after_shape = tuple(x.shape) if x is not None else None
    #                         print(f"[MACE-ATTN] {self.name} to_k: in={in_shape} after_proj={after_shape} expected_in={self.expected_in}")
    #                     except Exception:
    #                         pass
    #             except Exception:
    #                 pass
    #
    #             try:
    #                 if x is not None and self.expected_in is not None and x.shape[-1] != self.expected_in:
    #                     cur = x.shape[-1]
    #                     if cur > self.expected_in:
    #                         x = x[..., : self.expected_in]
    #                     else:
    #                         pad_shape = list(x.shape)
    #                         pad_shape[-1] = self.expected_in - cur
    #                         pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    #                         x = torch.cat([x, pad], dim=-1)
    #             except Exception:
    #                 pass
    #
    #             new_args = (x,) + rest
    #             return self.orig(*new_args, **f_kwargs)
    except Exception:
        global_text_proj = None

    # Ensure UNet receives a non-None added_cond_kwargs even if pipeline passes None.
    try:
        orig_unet_forward = pipe.unet.forward

        def _safe_unet_forward(*u_args, **u_kwargs):
            if u_kwargs.get("added_cond_kwargs") is None:
                u_kwargs["added_cond_kwargs"] = {}
            return orig_unet_forward(*u_args, **u_kwargs)

        pipe.unet.forward = _safe_unet_forward
    except Exception:
        # If for any reason we can't wrap, continue without failing — earlier checks still help.
        pass
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    torch.Generator(device=args.device).manual_seed(42)

    if args.generate_training_data:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        num_images = 8
        count = 0
        for single_concept in args.multi_concept:
            for c, t in single_concept:
                count += 1
                print(f"Generating training data for concept {count}: {c}...")
                c = c.replace('-', ' ')
                output_folder = f"{args.output_dir}/{c}"
                os.makedirs(output_folder, exist_ok=True)
                if t == "object":
                    prompt = f"a photo of the {c}"
                    print(f'Inferencing: {prompt}')
                    added_cond_kwargs = {}
                    # Always compute text embeddings and pass them to the pipeline.
                    # Some UNet/LoRA combinations expect explicit `text_embs` via
                    # `added_cond_kwargs` even when `addition_embed_type` is 'none'.
                    try:
                        tok = pipe.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
                        input_ids = tok.input_ids.to(pipe.text_encoder.device)
                        txt_out = pipe.text_encoder(input_ids)
                        # Extract tensor robustly
                        if isinstance(txt_out, (list, tuple)):
                            text_embs = txt_out[0]
                        elif hasattr(txt_out, "last_hidden_state"):
                            text_embs = getattr(txt_out, "last_hidden_state")
                        elif isinstance(txt_out, dict) and "text_embs" in txt_out:
                            text_embs = txt_out["text_embs"]
                        else:
                            text_embs = None
                        if text_embs is not None:
                            # Project to expected dim if any per-processor or global proj exists.
                            try:
                                # Prefer global simple projection if available
                                if global_text_proj is not None:
                                    text_embs = text_embs.to(next(global_text_proj.parameters()).device)
                                    text_embs = global_text_proj(text_embs)
                                else:
                                    # Try to use general pipe.unet.config.cross_attention_dim
                                    target_dim = getattr(pipe.unet.config, "cross_attention_dim", None)
                                    if target_dim is None:
                                        # fallback to inspecting attn_processors first match
                                        try:
                                            for _, p in getattr(pipe.unet, "attn_processors", {}).items():
                                                tk = getattr(p, "to_k", None)
                                                if isinstance(tk, torch.nn.Module):
                                                    w = getattr(tk, "weight", None)
                                                    if w is not None:
                                                        target_dim = w.shape[1]
                                                        break
                                        except Exception:
                                            target_dim = None
                                    if target_dim is not None and text_embs.shape[-1] != target_dim:
                                        proj = nn.Linear(text_embs.shape[-1], target_dim, bias=True).to(pipe.unet.device)
                                        with torch.no_grad():
                                            proj.weight.zero_()
                                            proj.bias.zero_()
                                            overlap = min(text_embs.shape[-1], target_dim)
                                            for d in range(overlap):
                                                proj.weight[d, d] = 1.0
                                        proj.requires_grad_(False)
                                        text_embs = text_embs.to(next(proj.parameters()).device)
                                        text_embs = proj(text_embs)
                            except Exception:
                                try:
                                    text_embs = text_embs.to(pipe.unet.device)
                                except Exception:
                                    pass
                            try:
                                added_cond_kwargs["text_embs"] = text_embs.to(pipe.unet.device)
                            except Exception:
                                added_cond_kwargs["text_embs"] = text_embs
                    except Exception:
                        added_cond_kwargs = {}
                    images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images, added_cond_kwargs=added_cond_kwargs).images
                    save_images(images, output_folder, prompt)
                    del images
                    torch.cuda.empty_cache()
                    gc.collect()
                elif t == "style":
                    prompt = f"a photo in the style of {c}"
                    print(f'Inferencing: {prompt}')
                    added_cond_kwargs = {}
                    # Always compute text embeddings and pass them to the pipeline.
                    # Some UNet/LoRA combinations expect explicit `text_embs` via
                    # `added_cond_kwargs` even when `addition_embed_type` is 'none'.
                    try:
                        tok = pipe.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
                        input_ids = tok.input_ids.to(pipe.text_encoder.device)
                        txt_out = pipe.text_encoder(input_ids)
                        if isinstance(txt_out, (list, tuple)):
                            text_embs = txt_out[0]
                        elif hasattr(txt_out, "last_hidden_state"):
                            text_embs = getattr(txt_out, "last_hidden_state")
                        elif isinstance(txt_out, dict) and "text_embs" in txt_out:
                            text_embs = txt_out["text_embs"]
                        else:
                            text_embs = None
                        if text_embs is not None:
                            try:
                                if global_text_proj is not None:
                                    text_embs = text_embs.to(next(global_text_proj.parameters()).device)
                                    text_embs = global_text_proj(text_embs)
                            except Exception:
                                text_embs = text_embs.to(pipe.unet.device)
                            added_cond_kwargs["text_embs"] = text_embs.to(pipe.unet.device)
                    except Exception:
                        added_cond_kwargs = {}
                    images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images, added_cond_kwargs=added_cond_kwargs).images
                    save_images(images, output_folder, prompt)
                else:
                    raise ValueError("unknown concept type.")
                del images
                torch.cuda.empty_cache()
                gc.collect()
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        num_images = args.num_images
        output_folder = f"{args.output_dir}/generated_images"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Inference using {args.pretrained_model_name_or_path}...")
        prompt = args.prompt
        added_cond_kwargs = {}
        # Always compute text embeddings and pass them to the pipeline.
        # Some UNet/LoRA combinations expect explicit `text_embs` via
        # `added_cond_kwargs` even when `addition_embed_type` is 'none'.
        try:
            tok = pipe.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
            input_ids = tok.input_ids.to(pipe.text_encoder.device)
            txt_out = pipe.text_encoder(input_ids)
            if isinstance(txt_out, (list, tuple)):
                text_embs = txt_out[0]
            elif hasattr(txt_out, "last_hidden_state"):
                text_embs = getattr(txt_out, "last_hidden_state")
            elif isinstance(txt_out, dict) and "text_embs" in txt_out:
                text_embs = txt_out["text_embs"]
            else:
                text_embs = None
            if text_embs is not None:
                try:
                    if global_text_proj is not None:
                        text_embs = text_embs.to(next(global_text_proj.parameters()).device)
                        text_embs = global_text_proj(text_embs)
                except Exception:
                    text_embs = text_embs.to(pipe.unet.device)
                added_cond_kwargs["text_embs"] = text_embs.to(pipe.unet.device)
        except Exception:
            added_cond_kwargs = {}
        images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images, added_cond_kwargs=added_cond_kwargs).images
        save_images(images, output_folder, prompt)
        del images
        torch.cuda.empty_cache()
        gc.collect()

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_images', type=int, default=3)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = 30
    model_id = args.model_path
    output_dir = args.save_path
    num_images = args.num_images
    prompt = args.prompt
    
    main(OmegaConf.create({
        "pretrained_model_name_or_path": model_id,
        "generate_training_data": False,
        "device": device,
        "steps": steps,
        "output_dir": output_dir,
        "num_images": num_images,
        "prompt": prompt,
    }))