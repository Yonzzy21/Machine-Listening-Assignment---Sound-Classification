# Sound Event Classification with Deep Learning
### CS-GY 6933: Machine Listening — Spring 2026

##  Project Overview
This project explores the fundamentals of deep learning applied to machine listening, as a part of Machine Listening Course at NYU, taught by Juan Pablo Bello.
The goal is to classify environmental audio recordings from the **ESC-50 dataset** using various neural network architectures. The pipeline includes raw audio processing, feature extraction via Mel-Frequency Cepstral Coefficients (MFCCs), and model optimization.

## Key Learning Objectives
* **Audio Engineering**: Using waveforms and spectrograms features.
* **Feature Extraction**: Implementing and tuning MFCC design for classification tasks.
* **Deep Learning Frameworks**: Building end-to-end pipelines in **PyTorch**, including custom `Datasets` and `DataLoaders`.
* **CNN Architectures**: Comparing the performance of 1D Convolutional Neural Networks (temporal) vs. 2D Convolutional Neural Networks (spectral).

##  Repository Structure
The assignment is organized into four logical parts:

| Section | Focus | Description |
| :--- | :--- | :--- |
| **Part 0** | **Tutorial** | Foundations of PyTorch: Tensors, autograd, and simple linear models. |
| **Part 1** | **MFCC Design** | Feature engineering and visualization of the ESC-50 dataset. |
| **Part 2** | **1D CNN Design** | Building a temporal model (190K parameters) for sound classification. |
| **Part 3** | **2D CNN & Optimization** | Leveraging 2D kernels for spectral images and Batch Normalization. |

##  Requirements & Setup
This project is designed to be executed in **Google Colab** with GPU acceleration (T4 or L4 runtimes).
