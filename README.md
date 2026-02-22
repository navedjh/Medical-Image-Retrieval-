# Medical-Image-Retrieval-
Semantic Image Retrieval System for PneumoniaMNIST using BioViL-T and FAISS
# Medical Image Retrieval System for PneumoniaMNIST

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navedjh/medical-image-retrieval/blob/main/notebooks/imageretrieval_task3.ipynb)

## 📋 Overview
This project implements a **semantic image retrieval system** for medical chest X-rays using the PneumoniaMNIST dataset. The system supports both image-to-image and text-to-image search with **83.5% precision@1**.

## ✨ Features
- 🖼️ **Image-to-Image Search**: Find visually similar chest X-rays
- 📝 **Text-to-Image Search**: Retrieve images using medical descriptions
- 🚀 **FAISS Vector Database**: Fast similarity search on 4,708 images
- 🏥 **BioViL-T Integration**: Medical text understanding
- 📊 **Precision@k Evaluation**: Quantitative performance metrics
- 🎨 **Visualization**: Color-coded results (green=correct, red=incorrect)

## 📊 Performance
| Metric | Value |
|--------|-------|
| **P@1** | **83.49%** |
| P@3 | 82.59% |
| P@5 | 81.60% |
| P@10 | 80.45% |

## 🏗️ Architecture: We propose a Hybrid Cross-Modal Retrieval System for chest X-ray analysis that integrates a supervised convolutional image encoder with a pretrained biomedical vision-language model. The system enables:

Image-to-Image retrieval

Text-to-Image retrieval

Clinically interpretable explainability

The framework consists of four primary components:

A CNN-based image encoder (ProvenCNN)

A pretrained biomedical text encoder (BioViL-T)

A shared normalized embedding space

A FAISS-based similarity search engine

The overall architecture is shown in Figure 1
<img width="1790" height="1190" alt="image" src="https://github.com/user-attachments/assets/6f9e3fa2-232c-4878-af49-69ea7b17b7d7" />

*Figure 1: Complete system architecture showing data flow from query input to retrieval results. The system achieves 83.5% precision@1 using a custom CNN for images and BioViL-T for text, with FAISS for similarity search.*


## 🚀 Quick Start

### Option 1: Run in Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/medical-image-retrieval/blob/main/notebooks/PneumoniaMNIST_Demo.ipynb)

### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/medical-image-retrieval.git
cd medical-image-retrieval

# Install dependencies
pip install -r task3_retrieval/requirements.txt

# Run the system
python -c "from task3_retrieval.retrieval_system import run_complete_system; run_complete_system()"

Evaluation
The system achieves 83.5% precision@1 on the PneumoniaMNIST test set, demonstrating excellent performance for medical image retrieval.

Clinical Relevance
With >80% precision across all metrics, this system is suitable for clinical decision support applications requiring similar case retrieval.

Author
Dr. Naved Alam

Acknowledgments
Microsoft for BioViL-T model

MedMNIST for dataset

FAISS for vector search library
