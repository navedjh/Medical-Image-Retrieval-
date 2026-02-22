"""
BioViL-T Adapter for PneumoniaMNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from PIL import Image

class BioViLTTextEncoder:
    """Text encoder using BioViL-T"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        model_url = "microsoft/BiomedVLP-BioViL-T"
        self.tokenizer = AutoTokenizer.from_pretrained(model_url, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_url, trust_remote_code=True).to(device)
        self.model.eval()
        self.embedding_dim = 128
    
    def encode(self, texts):
        inputs = self.tokenizer(texts, padding='longest', return_tensors='pt', 
                               truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.get_projected_text_embeddings(
                input_ids=input_ids, attention_mask=attention_mask
            )
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.cpu().numpy()


class ImageEncoderCNN(nn.Module):
    """CNN for encoding PneumoniaMNIST images"""
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.projection = nn.Linear(128, embedding_dim)
        
    def forward(self, x):
        features = self.cnn(x).view(x.size(0), -1)
        return F.normalize(self.projection(features), dim=-1)
    
    def encode(self, images, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.eval()
        with torch.no_grad():
            return self.forward(images.to(device)).cpu().numpy()


class UnifiedAdapter:
    """Combines BioViL-T for text and CNN for images"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.text_encoder = BioViLTTextEncoder(device)
        self.image_encoder = ImageEncoderCNN(self.text_encoder.embedding_dim).to(device)
        self.image_encoder.eval()
        self.embedding_dim = self.text_encoder.embedding_dim
    
    def encode_images(self, images):
        return self.image_encoder.encode(images, self.device)
    
    def encode_texts(self, texts):
        return self.text_encoder.encode(texts)