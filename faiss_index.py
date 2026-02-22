"""
FAISS index manager for similarity search
"""

import faiss
import numpy as np
import pickle
from pathlib import Path

class FAISSIndexManager:
    def __init__(self, dimension=128):
        self.dimension = dimension
        self.index = None
        self.metadata = {'indices': [], 'labels': []}
    
    def build_index(self, embeddings, labels, indices=None):
        print(f"\n🔨 Building FAISS index with {len(embeddings)} vectors...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        
        self.metadata['labels'] = labels.tolist() if isinstance(labels, np.ndarray) else labels
        if indices is not None:
            self.metadata['indices'] = indices.tolist() if isinstance(indices, np.ndarray) else indices
        else:
            self.metadata['indices'] = list(range(len(embeddings)))
        
        print(f"  ✓ Index built with {self.index.ntotal} vectors")
        return self.index
    
    def search(self, query_embedding, k=5):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.metadata['labels']):
                results.append({
                    'rank': i+1, 'score': float(score),
                    'index': int(self.metadata['indices'][idx]),
                    'label': int(self.metadata['labels'][idx])
                })
        return results
    
    def save(self, path):
        Path(path).mkdir(exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"  ✓ Index saved to {path}")
    
    def load(self, path):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"  ✓ Index loaded from {path} with {self.index.ntotal} vectors")
        return self.index