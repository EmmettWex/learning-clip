import glob
import torch
import torchvision
import math
import time
from annoy import AnnoyIndex
from PIL import Image
from sentence_transformers import SentenceTransformer

image_model = SentenceTransformer('clip-ViT-B-32')

images = []
image_paths = glob.glob('dataset/*_normals.png')

for path in image_paths:
    image = Image.open(path)
    images.append(image)

embeddings = image_model.encode(images)

COUNT = embeddings.shape[0]
LENGTH = embeddings.shape[1]

trees = 72

annoy_tree = AnnoyIndex(512, 'euclidean')
for idx, image in enumerate(embeddings):
    annoy_tree.add_item(idx, image)

annoy_tree.build(trees)
annoy_tree.save('images.ann')

query = AnnoyIndex(512, 'euclidean')
query.load('images.ann')
print(query.get_nns_by_item(0, 5, search_k=1000))