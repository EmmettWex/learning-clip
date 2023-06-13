import numpy as np
import os
import logging
import torch
import glob
import time
from tqdm import tqdm
import multiprocessing as mp
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from clip_retrieval.clip_client import ClipClient, Modality
from annoy import AnnoyIndex
from tinydb import TinyDB, Query
from typing import Dict, Optional


class VectorTinyDB:
    def __init__(self, dims, metric='angular', db_path='vector_tinydb', version='v1', verbose=False):
        self.metric = metric
        self.version = version
        self.db_path = db_path
        self.dims = dims
        self.vector_index = None
        self.tinydb = TinyDB(f'{self.db_path}_{self.version}.json')
        self.vector_db_path = f"{self.db_path}_{self.version}.ann"
        self.vectors = []
        self.verbose = verbose
        self.logger = self._get_logger(verbose)

    def _get_logger(self, verbose):
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Only show critical errors when not verbose
        logger.setLevel(logging.DEBUG if verbose else logging.CRITICAL)
        return logger

    def insert(self, vector, doc: Dict):
        doc_id = self.tinydb.insert(doc)
        self.vectors.append((doc_id, vector))
        self.logger.debug(f"Inserted document {doc_id}")

    def build_index(self):
        self.logger.debug("Building index...")
        self.vector_index = AnnoyIndex(self.dims, self.metric)
        for i, vector in self.vectors:
            self.vector_index.add_item(i, vector)
        self.vector_index.build(100)
        self.save_index()

    def save_index(self):
        self.vector_index.save(self.vector_db_path)
        self.logger.debug(f"Vector index saved to {self.vector_db_path}")

    def search(self, vector, n=1):
        if self.vector_index is None or len(self.vectors) > self.vector_index.get_n_items():
            self.logger.info("Index is out of date. Rebuilding...")
            self.build_index()
        nearest_ids = self.vector_index.get_nns_by_vector(vector, n)
        return [self.tinydb.get(doc_id=id) for id in nearest_ids]


## Collecting the dataset
# declare the model, processor, tokenizer for CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# declare the clip client
client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14", num_images=40)

# helper function to grab an array of descriptions from laion
def grab_descriptions(image):
    results = client.query(image=image)
    return results

def main():
    # using time module to benchmark
    start = time.time()

    # grab the image paths from the dataset
    image_paths = glob.glob('dataset/*_rgb.png')
    
    with mp.Pool(processes=8) as pool:
        processed_results = []
        index = 0
        for result in tqdm(pool.imap_unordered(grab_descriptions, image_paths)):
            processed_results.append(result)
            print(str(index))
            index += 1
        
    # Instantiate the database
    db = VectorTinyDB(dims=512, metric='euclidean', db_path='image_vector_db', version='v1', verbose=True)

    images = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
        
    inputs = processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        
    for idx, vector in enumerate(embeddings):
        db.insert(vector, {'data': processed_results[idx]})

    db.build_index()

    end = time.time()
    
    print(end - start)
    
    # Query the database with an arbitrary color vector
    # query_vector = rgb_to_vector((90, 60, 90))  # some kind of purple
    # result = db.search(query_vector, n=100)

if __name__ == '__main__':
    main()