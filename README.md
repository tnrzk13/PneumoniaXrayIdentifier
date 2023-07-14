# Pneumonia Detection with FastAI and Gradio: A Jupyter Notebook

This project demonstrates the development of a deep learning model that diagnoses pneumonia from chest X-ray images with an accuracy of 98.5%. The model is built using the FastAI library and a Gradio interface is created for easy use of the model. Due to the size of the xray dataset, it will not be included in this repo. However, if you would like to use the dataset for running this repo, it is available at https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The Jupyter notebook of the project is structured as follows:

## 1. Import

Here, all the required libraries and dependencies are imported. This includes the FastAI library for building the deep learning model and Gradio library for creating the user interface.

## 2. Load dataset

The chest X-ray images used in this project are sourced from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and labelled either 'pneumonia' or 'normal' (indicating no pneumonia). As part of the data cleaning process, null images are identified and removed.

## 3. Create a dataloader

In this section, a dataloader is created to manage the process of loading data into the model for both training and validation. The dataset is split such that 20% is reserved for validation. All images are resized to 128x128 for uniformity.

## 4. Training

The deep learning model is trained on the dataset. This is where the model learns to differentiate between 'normal' and 'pneumonia' chest X-ray images.

## 5. Analysis

The performance of the model is then analyzed. It was observed that the model has an error rate of less than 3%, which is impressive. Additional insights are drawn from the confusion matrix and top losses to further understand the model's performance and identify areas for potential improvement.

## 6. Data processing (again)

Here, any additional data processing steps that might be necessary are performed.

## 7. Create an interface using Gradio

A Gradio interface for the model is created in this section. This section can be used independently, provided a .pkl model has already been trained. This allows users to load the model and create the Gradio interface without going through the previous steps.

## 8. Export

The trained model and interface are then exported to a Python file. This file can be integrated into an application to use the pneumonia detection model independently.

This Jupyter notebook provides a complete guide to building a deep learning model using FastAI and creating a user interface for it using Gradio. The model achieves an accuracy of 98.5% in diagnosing pneumonia from chest X-ray images, making it a potentially powerful tool in the medical field.

