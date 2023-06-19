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
    ### class variables
    # declare the model, processor, tokenizer for CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # declare the clip client
    client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14", num_images=1)
    
    ### Class methods
    @staticmethod
    def grab_descriptions(image):
        results = VectorTinyDB.client.query(image=image)
        return results
    
    def __init__(self, dims, metric='angular', db_path='vector_tinydb', version='v1', verbose=False):
        self.metric = metric
        self.version = version
        self.db_path = db_path
        self.dims = dims
        self.tinydb = TinyDB(f'{self.db_path}_{self.version}.json')
        self.vector_db_path = f"{self.db_path}_{self.version}.ann"
        self.vector_index = None
        self.vectors = []
        
        # checking for the annoy index will assign the vector index and pre-populate
        # the vectors list if the index exists already
        self.check_for_annoy_index()
        self.verbose = verbose
        self.logger = self._get_logger(verbose)
        
    def check_for_annoy_index(self):
        if os.path.exists(self.vector_db_path):
            index = AnnoyIndex(self.dims, self.metric)
            index.load(self.vector_db_path)
            self.vector_index = index
            self.vectors = [self.vector_index.get_item_vector(i) for i in range(self.vector_index.get_n_items())]
        else:
            pass
        
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
        # if self.vector_index is None or len(self.vectors) > self.vector_index.get_n_items():
        #     self.logger.info("Index is out of date. Rebuilding...")
        #     self.build_index()
        nearest_ids = self.vector_index.get_nns_by_vector(vector, n, search_k=-1)
        
        print(nearest_ids)
        return [self.tinydb.get(doc_id=id) for id in nearest_ids]


### Collecting the dataset
## declare the model, processor, tokenizer for CLIP
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

## declare the clip client
# client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14", num_images=40)

# helper function to grab an array of descriptions from laion
# def grab_descriptions(image):
#     results = VectorTinyDB.client.query(image=image)
#     return results

def main():
    # using time module to benchmark
    start = time.time()

    # grab the image paths from the dataset
    image_paths = glob.glob('dataset/*_rgb.png')[:20]
    
    with mp.Pool(processes=8) as pool:
        processed_results = list(pool.imap(VectorTinyDB.grab_descriptions, image_paths))
        
    # Instantiate the database
    db = VectorTinyDB(dims=512, metric='euclidean', db_path='image_vector_db', version='v1', verbose=True)

    images = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
        
    inputs = VectorTinyDB.processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        embeddings = VectorTinyDB.model.get_image_features(**inputs)
        
    for idx, vector in enumerate(embeddings):
        db.insert(vector, {'data': processed_results[idx], 'source': image_paths[idx]}) # this dictionary will eventually also include a 'source_type', empty for now

    db.build_index()

    end = time.time()
    print(end - start)
    
    # Query the database with an arbitrary color vector
    # query_vector = rgb_to_vector((90, 60, 90))  # some kind of purple
    # result = db.search(query_vector, n=100)
    
def search(text_query="a cylinder"):
    # Instantiate the database
    db = VectorTinyDB(dims=512, metric='euclidean', db_path='image_vector_db', version='v1', verbose=True)
    
    input = VectorTinyDB.tokenizer([text_query], padding=True, return_tensors="pt")
    with torch.no_grad():
        vector = VectorTinyDB.model.get_text_features(**input)
    # print(vector[0])
    result = db.search(vector=vector[0])
    
    print(result)

def vectors():
    db = VectorTinyDB(dims=512, metric='euclidean', db_path='image_vector_db', version='v1', verbose=True)
    print(db.tinydb)

if __name__ == '__main__':
    # main()
    search()
    # vectors()