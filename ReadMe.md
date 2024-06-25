# Lung Cancer Detection and Staging

This repository contains the code for a hybrid model that integrates transformer-based models with graph neural networks (GNNs) for lung cancer detection and staging using CT scans. The project uses mixed precision training and is optimized for high-performance clusters.

## Introduction
This project aims to create a novel approach for lung cancer detection and staging by leveraging the strengths of both transformer models and graph neural networks (GNNs). The transformer captures global context, while the GNN focuses on local features.

## Dataset
The dataset used for this project is the Lung CT dataset with corresponding segmentation masks. You can download the dataset here: https://drive.google.com/file/d/1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi/view?usp=drive_link

The dataset directory structure is as follows:

Task06_Lung/
├── imagesTr
│   ├── lung_001.nii.gz
│   ├── lung_002.nii.gz
│   └── …
├── labelsTr
│   ├── lung_001.nii.gz
│   ├── lung_002.nii.gz
│   └── …
└── …

## Requirements
To install the required packages, run:
pip install -r requirements.txt
