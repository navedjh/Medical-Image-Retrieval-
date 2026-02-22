# ============================================
# COMPLETE PNEUMONIAMNIST RETRIEVAL SYSTEM
# ============================================

# Install required packages
!pip install medmnist faiss-cpu matplotlib tqdm -q

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import medmnist
from medmnist import INFO
import numpy as np
import faiss
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================
# PART 1: BIOVIL-T TEXT ENCODER (WORKING)
# ============================================

class BioViLTTextEncoder:
    """Text encoder using BioViL-T - produces 128-dim embeddings"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"\n🔤 Initializing BioViL-T text encoder...")
        
        # Load model and tokenizer
        model_url = "microsoft/BiomedVLP-BioViL-T"
        self.tokenizer = AutoTokenizer.from_pretrained(model_url, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_url, trust_remote_code=True).to(device)
        self.model.eval()
        
        # From exploration, projection_size = 128
        self.embedding_dim = 128
        print(f"  ✓ Text encoder ready. Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts):
        """
        Encode texts into normalized embeddings
        
        Args:
            texts: list of strings
        Returns:
            embeddings: numpy array (len(texts), 128) normalized
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=128
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # Use the specific method for projected embeddings
            embeddings = self.model.get_projected_text_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Normalize for cosine similarity
            embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings.cpu().numpy()


# ============================================
# PART 2: CNN IMAGE ENCODER (FOR PNEUMONIAMNIST)
# ============================================

class ImageEncoderCNN(nn.Module):
    """CNN for encoding PneumoniaMNIST images into 128-dim embeddings"""
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # CNN architecture specifically for 28x28 medical images
        self.cnn = nn.Sequential(
            # First conv block: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block: 7x7 -> features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling -> 1x1
        )
        
        # Projection layer to match BioViL-T dimension
        self.projection = nn.Linear(128, embedding_dim)
        
    def forward(self, x):
        """Forward pass - returns normalized embeddings"""
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1)
    
    def encode(self, images, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Encode images into normalized embeddings"""
        self.eval()
        with torch.no_grad():
            images = images.to(device)
            embeddings = self.forward(images)
        return embeddings.cpu().numpy()


# ============================================
# PART 3: UNIFIED ADAPTER
# ============================================

class UnifiedAdapter:
    """Combines BioViL-T for text and CNN for images"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print("\n" + "="*60)
        print("UNIFIED ADAPTER")
        print("="*60)
        
        # Initialize text encoder (BioViL-T)
        self.text_encoder = BioViLTTextEncoder(device)
        
        # Initialize image encoder (Custom CNN)
        self.image_encoder = ImageEncoderCNN(embedding_dim=self.text_encoder.embedding_dim)
        self.image_encoder = self.image_encoder.to(device)
        self.image_encoder.eval()
        
        self.embedding_dim = self.text_encoder.embedding_dim
        print(f"\n✓ Unified adapter ready")
        print(f"  - Text encoder: BioViL-T (128-dim)")
        print(f"  - Image encoder: Custom CNN (128-dim)")
        print(f"  - Total parameters: {sum(p.numel() for p in self.image_encoder.parameters()):,}")
    
    def encode_images(self, images):
        """Encode images using CNN"""
        return self.image_encoder.encode(images, self.device)
    
    def encode_texts(self, texts):
        """Encode texts using BioViL-T"""
        return self.text_encoder.encode(texts)


# ============================================
# PART 4: PNEUMONIAMNIST DATASET
# ============================================

class PneumoniaMNISTDataset(Dataset):
    """Wrapper for PneumoniaMNIST dataset"""
    
    def __init__(self, split='train'):
        self.dataset = medmnist.PneumoniaMNIST(split=split, download=True)
        self.split = split
        self.label_names = {0: 'Normal', 1: 'Pneumonia'}
        print(f"  Loaded {split} set: {len(self.dataset)} images")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Convert PIL to tensor: (28, 28) -> (1, 28, 28), values in [0,1]
        img_array = np.array(img)
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0) / 255.0
        label_value = label[0]  # Label is [0] or [1]
        return img_tensor, label_value, idx
    
    def get_pil_image(self, idx):
        """Get original PIL image for visualization"""
        img, label = self.dataset[idx]
        return img, label[0]


def create_dataloader(dataset, batch_size=64, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================
# PART 5: FAISS INDEX MANAGER
# ============================================

class FAISSIndexManager:
    """Manages FAISS index for similarity search"""
    
    def __init__(self, dimension=128):
        self.dimension = dimension
        self.index = None
        self.metadata = {'indices': [], 'labels': []}
    
    def build_index(self, embeddings, labels, indices=None):
        """Build FAISS index from embeddings"""
        print(f"\n🔨 Building FAISS index...")
        print(f"  - Vectors: {len(embeddings)}")
        print(f"  - Dimension: {self.dimension}")
        
        # Use flat index (exact search) for simplicity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata['labels'] = labels.tolist() if isinstance(labels, np.ndarray) else labels
        self.metadata['indices'] = indices if indices is not None else list(range(len(embeddings)))
        
        print(f"  ✓ Index built with {self.index.ntotal} vectors")
        return self.index
    
    def search(self, query_embedding, k=5):
        """Search for k nearest neighbors"""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.metadata['labels']):
                results.append({
                    'rank': i + 1,
                    'score': float(score),
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


# ============================================
# PART 6: MAIN RETRIEVAL SYSTEM
# ============================================

class PneumoniaRetrievalSystem:
    """Complete retrieval system for PneumoniaMNIST"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print("\n" + "="*60)
        print("PNEUMONIAMNIST RETRIEVAL SYSTEM")
        print("="*60)
        
        self.adapter = UnifiedAdapter(device)
        self.embedding_dim = self.adapter.embedding_dim
        self.index_manager = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.label_names = {0: 'Normal', 1: 'Pneumonia'}
    
    def load_datasets(self):
        """Load all dataset splits"""
        print("\n📁 Loading datasets...")
        self.train_dataset = PneumoniaMNISTDataset('train')
        self.val_dataset = PneumoniaMNISTDataset('val')
        self.test_dataset = PneumoniaMNISTDataset('test')
    
    def extract_image_embeddings(self, dataset, desc="Extracting embeddings", batch_size=64):
        """Extract image embeddings for entire dataset"""
        dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
        
        all_embeddings = []
        all_labels = []
        all_indices = []
        
        for images, labels, indices in tqdm(dataloader, desc=desc):
            embeddings = self.adapter.encode_images(images)
            all_embeddings.append(embeddings)
            all_labels.extend(labels.numpy())
            all_indices.extend(indices.numpy())
        
        return np.vstack(all_embeddings), np.array(all_labels), np.array(all_indices)
    
    def build_index(self, save_path='./pneumonia_index'):
        """Build retrieval index from training set"""
        if self.train_dataset is None:
            self.load_datasets()
        
        print("\n🔧 Building index from training set...")
        embeddings, labels, indices = self.extract_image_embeddings(self.train_dataset)
        
        print(f"\n📊 Statistics:")
        print(f"  - Embedding shape: {embeddings.shape}")
        print(f"  - Normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")
        
        self.index_manager = FAISSIndexManager(embeddings.shape[1])
        self.index_manager.build_index(embeddings, labels, indices)
        self.index_manager.save(save_path)
        return self.index_manager
    
    def load_index(self, path='./pneumonia_index'):
        """Load existing index"""
        self.index_manager = FAISSIndexManager()
        self.index_manager.load(path)
    
    def search_by_image(self, query_idx, k=5, split='test'):
        """Search using image by index"""
        if self.index_manager is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        # Get the right dataset
        if split == 'test':
            dataset = self.test_dataset
        elif split == 'val':
            dataset = self.val_dataset
        else:
            dataset = self.train_dataset
        
        if dataset is None:
            self.load_datasets()
            dataset = self.test_dataset if split == 'test' else self.val_dataset
        
        # Get query image
        query_img_tensor, query_label, _ = dataset[query_idx]
        query_embedding = self.adapter.encode_images(query_img_tensor.unsqueeze(0))
        results = self.index_manager.search(query_embedding, k)
        
        # Add label names
        for r in results:
            r['label_name'] = self.label_names[r['label']]
        
        # Get PIL image for visualization
        query_pil, _ = dataset.get_pil_image(query_idx)
        query_info = {
            'index': query_idx,
            'label': int(query_label),
            'label_name': self.label_names[int(query_label)],
            'image': query_pil
        }
        
        return query_info, results
    
    def search_by_text(self, text_query, k=5):
        """Search using text description"""
        if self.index_manager is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        text_embeddings = self.adapter.encode_texts([text_query])
        results = self.index_manager.search(text_embeddings, k)
        
        for r in results:
            r['label_name'] = self.label_names[r['label']]
        return results
    
    def evaluate(self, k_values=[1, 3, 5, 10], split='test'):
        """Evaluate retrieval performance using Precision@k"""
        if self.index_manager is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        # Get evaluation dataset
        if split == 'test':
            if self.test_dataset is None:
                self.load_datasets()
            dataset = self.test_dataset
        else:
            if self.val_dataset is None:
                self.load_datasets()
            dataset = self.val_dataset
        
        print(f"\n📊 Evaluating on {split} set ({len(dataset)} images)...")
        eval_embeddings, eval_labels, _ = self.extract_image_embeddings(dataset)
        
        results = {f'P@{k}': [] for k in k_values}
        
        for i in tqdm(range(len(eval_embeddings)), desc="  Processing queries"):
            query_emb = eval_embeddings[i:i+1]
            query_label = eval_labels[i]
            search_results = self.index_manager.search(query_emb, max(k_values))
            
            for k in k_values:
                if len(search_results) >= k:
                    relevant = sum(1 for r in search_results[:k] if r['label'] == query_label)
                    precision = relevant / k
                else:
                    precision = 0.0
                results[f'P@{k}'].append(precision)
        
        avg_results = {k: np.mean(v) for k, v in results.items()}
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for k, v in avg_results.items():
            print(f"  {k}: {v:.4f}")
        return avg_results
    
    def visualize_results(self, query_info, results, query_type='image', output_file=None):
        """Visualize query and top-k retrieved images"""
        k = len(results)
        fig, axes = plt.subplots(1, k+1, figsize=(3*(k+1), 4))
        
        if k == 0:
            axes = [axes]
        
        # Plot query
        if query_type == 'image':
            axes[0].imshow(np.array(query_info['image']), cmap='gray')
            axes[0].set_title(f"Query\n{query_info['label_name']}", fontweight='bold')
            query_label = query_info['label']
        else:
            axes[0].text(0.5, 0.5, f'"{query_info}"', ha='center', va='center', fontsize=10)
            axes[0].set_title("Text Query", fontweight='bold')
            query_label = None
        axes[0].axis('off')
        
        # Plot retrieved images
        for i, result in enumerate(results):
            if self.train_dataset is None:
                self.load_datasets()
            img, _ = self.train_dataset.get_pil_image(result['index'])
            axes[i+1].imshow(np.array(img), cmap='gray')
            
            # Color code: green = correct, red = incorrect
            if query_label is not None:
                color = 'green' if result['label'] == query_label else 'red'
            else:
                color = 'black'
            
            axes[i+1].set_title(f"Rank {i+1}\nScore: {result['score']:.3f}\n{result['label_name']}", 
                               color=color, fontsize=9)
            axes[i+1].axis('off')
        
        plt.suptitle(f"{query_type.capitalize()}-to-Image Retrieval Results", fontsize=14)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n  ✓ Visualization saved to {output_file}")
        plt.show()


# ============================================
# PART 7: RUN THE COMPLETE SYSTEM
# ============================================

def run_complete_system():
    """Run the complete retrieval system"""
    
    print("\n" + "="*60)
    print("🚀 STARTING COMPLETE RETRIEVAL SYSTEM")
    print("="*60)
    
    # Initialize system
    system = PneumoniaRetrievalSystem()
    
    # Step 1: Load datasets
    print("\n" + "-"*40)
    print("STEP 1: LOADING DATASETS")
    print("-"*40)
    system.load_datasets()
    
    # Step 2: Build index from training set
    print("\n" + "-"*40)
    print("STEP 2: BUILDING FAISS INDEX")
    print("-"*40)
    system.build_index()
    
    # Step 3: Test image-to-image search
    print("\n" + "-"*40)
    print("STEP 3: IMAGE-TO-IMAGE SEARCH")
    print("-"*40)
    query_idx = 42
    print(f"\n🔍 Query image index: {query_idx}")
    query_info, results = system.search_by_image(query_idx, k=5)
    print(f"   Query label: {query_info['label_name']}")
    print("\n   Retrieved images:")
    for r in results:
        print(f"     Rank {r['rank']}: {r['label_name']} (score={r['score']:.4f})")
    
    # Visualize
    system.visualize_results(query_info, results, query_type='image', output_file='image_search_results.png')
    
    # Step 4: Test text-to-image search
    print("\n" + "-"*40)
    print("STEP 4: TEXT-TO-IMAGE SEARCH")
    print("-"*40)
    text_queries = [
        "clear lungs no infiltrates",
        "pneumonia consolidation",
        "normal chest x-ray"
    ]
    
    for text_query in text_queries:
        print(f"\n📝 Query: '{text_query}'")
        results = system.search_by_text(text_query, k=5)
        for r in results:
            print(f"     Rank {r['rank']}: {r['label_name']} (score={r['score']:.4f})")
    
    # Visualize first text query
    system.visualize_results(text_queries[0], system.search_by_text(text_queries[0], 5), 
                            query_type='text', output_file='text_search_results.png')
    
    # Step 5: Evaluate system performance
    print("\n" + "-"*40)
    print("STEP 5: EVALUATION")
    print("-"*40)
    metrics = system.evaluate([1, 3, 5, 10])
    
    # Step 6: Summary
    print("\n" + "="*60)
    print("✅ SYSTEM COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\n📁 Output files:")
    print("  - image_search_results.png (image-to-image search)")
    print("  - text_search_results.png (text-to-image search)")
    print("  - ./pneumonia_index/ (FAISS index and metadata)")
    print("\n📊 Final Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return system


# ============================================
# RUN EVERYTHING
# ============================================

if __name__ == "__main__":
    system = run_complete_system()