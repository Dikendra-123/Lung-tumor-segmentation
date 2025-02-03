# Lung Tumor Segmentation
# Overview
This project aims to develop an accurate and efficient lung segmentation model for medical image analysis using the SegUNet architecture. The model is designed to segment lung tumor regions from CT scans, which is a critical step in diagnosing and treating pulmonary diseases. The model was trained on the Medical Decathlon competition dataset. 

# Key Features

SegUNet Architecture: Utilizes a combination of U-Net and segmentation techniques for precise lung tumor detection.
Medical Decathlon Dataset: Trained on a diverse and challenging dataset to ensure robustness across various imaging scenarios.
Efficient Processing: Optimized for high-performance inference, enabling real-time analysis of medical images.

# Dataset

The model was trained on the [Medical Decathlon competition](https://drive.google.com/file/d/1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi/view?usp=sharing)
dataset, which includes a wide range of annotated CT scans. This dataset is widely used in medical image analysis for its diversity and complexity, making it ideal for developing robust segmentation models.

# Model Architecture
The [SegUNet model](https://docs.google.com/document/d/1ngOpqQ98q2Mx6_RXanPms5LvOj6k4NrwkwPMZQ-WqeA/edit?usp=sharing) combines the strengths of U-Net with advanced segmentation techniques:
Encoder-Decoder Structure: Captures contextual information while preserving spatial details.
Skip Connections: Enhances feature propagation for accurate boundary detection.
<img width="596" alt="SegUNet Architecture" src="https://github.com/user-attachments/assets/d3dea966-6f03-4824-b622-30da0c8e9bc8" />

# Training setup
We trained single-contrast 3D CT scans of lung tumors from 130 different subjects (approximately 58,500 slices) using the Ubuntu 22.04 operating system with an NVIDIA RTX 4070 Ti GPU card. 

# output 
Sample output tumor segmentation images are given below:<img width="596" alt="output" src="https://github.com/user-attachments/assets/9a671ec7-19f9-429d-8fbf-d15cb90fa7b6" />
# 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

