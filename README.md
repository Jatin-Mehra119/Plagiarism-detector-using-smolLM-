---
license: mit
title: Plagiarism-detector-using-Fine-tuned-smolLM
sdk: streamlit
App link: https://jatinmehra-plagiarism-detector-using-smollm.hf.space/
Fine tuned Model: https://huggingface.co/jatinmehra/smolLM-fined-tuned-for-PLAGAIRISM_Detection
---

# Plagiarism Detection App Using a Fine-Tuned Language Model (LLM)

This repository contains a Streamlit-based web application that uses a fine-tuned LLM model for detecting plagiarism between two documents. The application processes two uploaded PDF files, extracts their content, and classifies them as either plagiarized or non-plagiarized based on a fine-tuned language model.

## Overview

The app leverages a **custom fine-tuned version of the SmolLM** (135M parameters) that has been trained on the [MIT Plagiarism Detection Dataset](https://www.kaggle.com/datasets/ruvelpereira/mit-plagairism-detection-dataset) for improved performance in identifying textual similarities. This model provides binary classification outputs, indicating if two given documents are plagiarized or original.

## Features

-   **Upload PDF Files**: Upload two PDF files that the app will analyze for similarity.
-   **Text Extraction**: Extracts raw text from the uploaded PDFs using PyMuPDF.
-   **Model-Based Detection**: Compares the content of the PDFs and classifies them as plagiarized or non-plagiarized using the fine-tuned language model.
-   **User-Friendly Interface**: Built with Streamlit for an intuitive and interactive experience.

## Model Information

-   **Base Model**: `HuggingFaceTB/SmolLM2-135M-Instruct`
-   **Fine-tuned Model Name**: `jatinmehra/smolLM-fine-tuned-for-plagiarism-detection`
-   **Language**: English
-   **Task**: Text Classification (Binary)
-   **Performance Metrics**: Accuracy, F1 Score, Recall
-   **License**: MIT

## Dataset

The fine-tuning dataset, the MIT Plagiarism Detection Dataset, provides labeled sentence pairs where each pair is marked as plagiarized or non-plagiarized. This label is used for binary classification, making it well-suited for detecting sentence-level similarity.

## Training and Model Details

-   **Architecture**: The model was modified for sequence classification with two labels.
-   **Optimizer**: AdamW with a learning rate of 2e-5.
-   **Loss Function**: Cross-Entropy Loss.
-   **Batch Size**: 16
-   **Epochs**: 3
-   **Padding**: Custom padding token to align with SmolLM requirements.

The model achieved **99.66% accuracy** on the training dataset, highlighting its effectiveness in identifying plagiarized content.

## Application Workflow

1.  **Load and Initialize**: The application loads the fine-tuned model and tokenizer locally.
2.  **PDF Upload**: Users upload two PDF documents they want to compare.
3.  **Text Extraction**: Text is extracted from each PDF using the PyMuPDF library.
4.  **Preprocessing**: The extracted text is tokenized and preprocessed for model compatibility.
5.  **Classification**: The model processes the inputs and returns a prediction of `1` (plagiarized) or `0` (non-plagiarized).
6.  **Output**: The result is displayed on the Streamlit interface.

## How to Run the Application

### Prerequisites

-   **Streamlit** for running the web application interface.
-   **Transformers** from Hugging Face for handling model and tokenizer.
-   **PyMuPDF** (`fitz`) for PDF text extraction.
-   **Torch** for model inference on CPU or GPU.

### Installation

1.  Clone the repository:
    
    bash
    
    Copy code
    
    `git clone https://github.com/YourUsername/Plagiarism-Detection-App.git
    cd Plagiarism-Detection-App` 
    
2.  Install the required dependencies:
    
    bash
    
    Copy code
    
    `pip install -r requirements.txt` 
    
3.  Download the fine-tuned model files and place them in the `model/` directory.
    
### Running the App

Run the Streamlit app from the terminal:

bash

Copy code

`streamlit run app.py` 

### Usage

1.  Open the application in your browser (default at `http://localhost:8501`).
2.  Upload two PDF files you wish to compare for plagiarism.
3.  View the text from each document and the resulting plagiarism detection output.

## Evaluation

The model was evaluated on both training and test data, showing robust results:

-   **Training Set Accuracy**: **99.66%**
-   **Test Set Accuracy**: **100%**
-   **F1 Score**: **1.0**
-   **Recall**: **1.0**

These metrics indicate the model's high effectiveness in detecting plagiarism.

## Model and Tokenizer

The model and tokenizer are saved locally, but they can also be loaded directly from Hugging Face. This setup allows easy loading for custom applications or further fine-tuning.

## License

This project is licensed under the MIT License, making it free for both personal and commercial use.

## Connect with Me

I appreciate your interest!  
[GitHub](https://github.com/Jatin-Mehra119) | Email-jatinmehra@outlook.in | [LinkedIn](https://www.linkedin.com/in/jatin-mehra119/) | [Portfolio](https://jatin-mehra119.github.io/Profile/)
