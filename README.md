# LoRA for SemanticSegmentation-domain-adaptation - CVC Internship

**Institution:** [Computer Vision Center (CVC)](https://www.cvc.uab.es/)
**Internship Period:** January 24 - June 24

## Project Overview

In this project, we focus on domain adaptation for semantic segmentation using the LoRA (Low-Rank Adaptation) technique. The primary objective is to improve the performance of semantic segmentation models when transferring knowledge from a synthetic dataset to a real-world dataset.

We use the SegFormer-B0 model, a state-of-the-art transformer-based architecture designed for efficient and accurate semantic segmentation. The project consists of two main phases:

1. **Pretraining on GTA5 Dataset:**
   - **Dataset:** GTA5
   - **Labels:** Cityscapes
   - **Description:** We pretrain the SegFormer-B0 model on the GTA5 dataset, which provides synthetic images with labels compatible with the Cityscapes dataset. This phase aims to leverage the large-scale synthetic data to learn robust feature representations for semantic segmentation.

2. **Domain Adaptation on Cityscapes Dataset:**
   - **Dataset:** Cityscapes
   - **Technique:** Low-Rank Adaptation (LoRA)
   - **Description:** After pretraining, we apply the LoRA technique to adapt the pretrained model to the Cityscapes dataset, which consists of real-world urban street scenes. The LoRA method enables efficient fine-tuning by introducing low-rank updates, reducing the number of trainable parameters and enhancing generalization capabilities.

This approach allows us to utilize the vast and diverse synthetic data from GTA5 to build a strong foundation for the model, followed by domain adaptation to bridge the gap between synthetic and real-world data. The expected outcome is a robust semantic segmentation model that performs well on real-world data with improved accuracy and efficiency.

By combining SegFormer-B0 with the LoRA technique, this project aims to advance the field of domain adaptation in semantic segmentation, providing valuable insights and methodologies for future research and applications.

## Content
### Main Scripts
- **[gta_segformer_trainer](scripts/gta_segformer_trainer.py):** Trains the **SegFormer-B0** on the **GTA** dataset and stores a checkpoint to the trained model.
- **[lora_gta_to_cityscapes](scripts/lora_gta_to_cityscapes_clean.py):** Takes a previously pretrained **SegFormer-B0** on **GTA** and performs **LoRA** domain adaptation to a **Cityscapes** dataset.
- **[evaluation](scripts/Segformer_evaluation.ipynb):** Performs several model evaluations:
  - Model: **SegFormer-B0** trained on **GTA** - Evaluation Dataset: **GTA**
  - Model: **SegFormer-B0** trained on **GTA** - Evaluation Dataset: **Cityscapes**
  - Model: **SegFormer-B0 LORA** trained on first **GTA** and then **Cityscapes** - Evaluation Dataset: **Cityscapes**  
### Secondary Scripts
- **[cityscapes_dataset_creator](useful_scripts/cityscapes_dataset_creator.py):** Given raw data, creates a **Cityscapes** dataset for further use.
- **[gta_dataset_creator](useful_scripts/gta_dataset_creator.py):** Given raw data, creates a **batched GTA** dataset for further use.
- **[cityscapes_dataset_creator](useful_scripts/cityscapes_dataset_creator.py):** Given a **batched GTA** dataset, loads it for further use.