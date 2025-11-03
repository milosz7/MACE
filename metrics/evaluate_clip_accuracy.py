import os
from PIL import Image
import pandas as pd
import re
import clip
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import json


@torch.no_grad()
def calculate_mean_prob(image_dir, object_ls, save_path, save_means):
    '''
    Returns a dataframe, where the first column is the image name, and the next ten columns are the clip_score with each object
    ------------------------------
    image_dir: Path to the image folder str
    object_ls: List of objects to classify [str, str, ...]
    '''
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    texts_ls=[f'a photo of the {object}' for object in object_ls]
    text_tokens = clip.tokenize(texts_ls).to(device)

    image_filenames=os.listdir(image_dir)
    sorted_image_filenames = sorted(image_filenames, key=lambda x: int(re.search(r'_(\d+)\.', x).group(1)))

    prob_results=[]
    for i in tqdm(range(len(sorted_image_filenames))):
        image_name=sorted_image_filenames[i]
        image = preprocess(Image.open(os.path.join(image_dir,image_name))).unsqueeze(0).to(device)
        
        image_features = model.encode_image(image).float()
        text_features = model.encode_text(text_tokens).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().tolist()[0]
    
        prob_result= {"ImageName": image_name,}
        for j, object in enumerate(object_ls):
            prob_result[object] = probs[j]
        prob_results.append(prob_result)

    prob_df = pd.DataFrame(prob_results)

    if save_path:
        prob_df.to_csv(f'{save_path}/{image_dir.replace("/","_")}.csv')
    else:
        print('do not need to save detailed probability results')
    
    caring_object= image_dir.split('/')[-1]
    if "_paraphrases" in caring_object:
        caring_object = caring_object.replace("_paraphrases", "")

    if save_means:
        prob_df.to_csv(f'{save_path}/{caring_object}.csv')
        caring_average_prob=prob_df[caring_object].mean()
    
        return caring_average_prob
    return 0.0


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--n_images_in_dir", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--save_means", type=bool, required=False, default=False)
    args = parser.parse_args()

    base_folder=args.base_folder
    save_path=args.save_path
    save_means=args.save_means

    with open('./prompts_csv/10_objects_paraphrase.json','r') as file:
        object_dic=json.load(file)
    object_ls=list(object_dic.keys())

    def process_folder_recursive(base_folder, n_files_in_dir):
        results = {}
        if len(os.listdir(base_folder)) == n_files_in_dir:        # end folder
            print('End folder is:', base_folder)
            result = calculate_mean_prob(base_folder, object_ls, save_path, save_means)
            results[base_folder] = result
        else:                                         # parent folder
            for folder_name in os.listdir(base_folder):
                folder_path = os.path.join(base_folder, folder_name)
                results.update(process_folder_recursive(folder_path, n_files_in_dir))
        return results

    all_results=process_folder_recursive(base_folder, args.n_images_in_dir)
    if save_means:
        all_results = {key.split("/")[-1]: str(value) for key, value in all_results.items()}
        all_results = [",".join(item) for item in all_results.items()]
        headers = ["category", "accuracy"]
        with open(f"{save_path}/all_results.csv","w") as file:
            file.write(",".join(headers))
            file.write("\n")
            file.write("\n".join(all_results))

