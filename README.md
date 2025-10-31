# Cattle Breed Detection using CNN

A deep learning project that classifies cattle images into 5 different breeds using Convolutional Neural Networks.

##  Project Overview

This is my first hands-on CNN project, built while learning MIT's Intro to Deep Learning course. The model achieves **85%+ accuracy** on the test set.

##  Dataset

- **Source:** [Kaggle - Cattle Breeds Dataset](https://www.kaggle.com/datasets/anandkumarsahu09/cattle-breeds-dataset)
- **Total Images:** 1,207
- **Number of Classes:** 5 cattle breeds
- **Split:** 80% training, 20% testing (stratified)

##  Model Architecture

- Custom CNN with 6 convolutional blocks
- Batch Normalization after each Conv layer
- Dropout for regularization
- Global Average Pooling
- Input size: 300×300×3
- Optimizer: Adam (learning rate: 0.0005)

##  Results

- **Training Accuracy:** ~88%
- **Test Accuracy:** ~85%
- Includes data augmentation to improve generalization

##  Tech Stack

- Python
- TensorFlow/Keras
- NumPy, Matplotlib
- Scikit-learn
- Google Colab

##  How to Use

1. Open `cattle_breed.ipynb` in Google Colab
2. Upload your `kaggle.json` API key when prompted
3. Run all cells
4. Model will be saved as `best_cattle_model.keras`

##  What I Learned

- Implementing CNNs from scratch
- Proper data splitting with stratification
- Data augmentation techniques
- Preventing overfitting with dropout and batch normalization
- Model evaluation and visualization

##  Related

Part of my Deep Learning learning journey following MIT 6.S191.



