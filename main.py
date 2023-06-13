import torch
import torchvision
# import math
# import time
import glob
from annoy import AnnoyIndex
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from clip_retrieval.clip_client import ClipClient, Modality

client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-H-14")

## creating the trees here
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# images = []
# image_paths = glob.glob('dataset/*_rgb.png')
cat_results = client.query(image='dataset/4937_rgb.png')
print(cat_results)
# print(image_paths)

# for path in image_paths:
#     image = Image.open(path)
#     images.append(image)

# inputs = processor(images=images, return_tensors="pt", padding=True)

# with torch.no_grad():
#     embeddings = model.get_image_features(**inputs)
    
# print(embeddings.shape)

# trees = 72

# annoy_tree = AnnoyIndex(512, 'euclidean')
# for idx, image in enumerate(embeddings):
#     annoy_tree.add_item(idx, image)

# annoy_tree.build(trees)
# annoy_tree.save('images.ann')

## query code below:
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# text_query = "a rendering of a steering wheel"

# input = tokenizer([text_query], padding=True, return_tensors="pt")
# with torch.no_grad():
#     vector = model.get_text_features(**input)
# print(vector.shape)

# query = AnnoyIndex(512, 'euclidean')
# query.load('images.ann')
# indexes = query.get_nns_by_vector(vector[0], 2, search_k=-1)
# results = [image_paths[i] for i in indexes]
# print(results)