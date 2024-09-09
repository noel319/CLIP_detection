import argparse
import os
import cv2
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import src.config as CFG
from src.train import build_loaders, make_train_valid_dfs
from src.CLIP import CLIPModel

def get_image_embeddings(valid_df, model_path):        
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

def find_matches(model, image_embeddings, query, image_filenames, n=1):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    _, indices = torch.topk(dot_similarity.squeeze(0),1)
    matches = image_filenames[indices[::1]]
    if matches == args.f:
        print("TRUE")
        
    else:
        print("False")    
    _, axes = plt.subplots(1, 1, figsize=(10, 10))
    image = cv2.imread(f"{CFG.image_path}/{matches}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes.imshow(image)
    axes.axis("off")        
    plt.show()
    if matches == args.f:
        print("TRUE")
        
    else:
        print("False")
if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Image Detection Source Intelligence Automation.')
    p.add_argument("-f", metavar="FILE", type=str, help="Image File URL")
    p.add_argument("-q", metavar="QUERY", type=str, help="Query of image file")
    valid_df = pd.read_csv('data/processed/train/captions.csv')
    valid_df = valid_df[:8]
    args = p.parse_args()
    
    if args.f and args.q:
        model, image_embeddings = get_image_embeddings(valid_df,  "models/best.pt")
        find_matches(model, image_embeddings, args.q, image_filenames=valid_df['image'].values, n=1)
    else:
        print(f"Input as this type:")
        print(f"py app.py -f 'Image URL' -q 'text query'")