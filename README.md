# Sentiment-Analysis-of-Hotel-Reviews-using-Fine-Tuned-BERT
Fine-tuned BERT model for sentiment analysis of hotel reviews using Hugging Face Transformers. Includes training, evaluation, and visualization of model performance.

This project performs sentiment analysis on hotel reviews by fine-tuning a BERT-based model using the Hugging Face Transformers library. The goal is to accurately classify reviews into **positive** or **negative** sentiments and evaluate model performance using key metrics and visualizations.

## 📌 Project Overview

Sentiment analysis is a crucial Natural Language Processing (NLP) task that identifies and categorizes sentiments expressed in textual data. This project focuses on hotel reviews, which are valuable for understanding customer satisfaction and service quality.

Unlike traditional approaches, this study utilizes a fine-tuned BERT model to improve accuracy and reduce misclassifications in sentiment predictions.

## 🧠 Model Used

- **BERT (Bidirectional Encoder Representations from Transformers)** – pre-trained transformer model from Hugging Face
- Fine-tuned on a labeled dataset of hotel reviews (positive or negative)


## 📊 Features

- Preprocessing of review text
- Tokenization using Hugging Face tokenizer
- Fine-tuning of BERT model
- Evaluation using classification report and accuracy
- Visualization of performance (bar chart)
- Exports predictions to CSV

## 🚀 Getting Started

### Requirements

- Python 3.7+
- Transformers
- Datasets
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch

Install dependencies with:

bash
pip install -r requirements.txt
How to Run
Place your train and test CSV files inside the data/ folder.

Run the training and evaluation:

bash
Copy
Edit
python fine_tune_bert_sentiment.py
To generate the accuracy bar chart:

bash
Copy
Edit
python plot_metrics.py
View your predictions in predicted_reviews.csv.

##📈 Results
The fine-tuned BERT model achieved:

Accuracy: 91.80%

F1-Score: 0.92 (macro average)

Misclassifications were minimal, as supported by the classification report and the visualized bar chart.

##📚 Acknowledgments
Hugging Face Transformers

PyTorch

Original idea inspired by the study: Optimization Techniques for Sentiment Analysis Based on LLM (GPT-3) [1]

##🔖 Keywords
Sentiment Analysis, BERT, NLP, Hotel Reviews, Hugging Face, Fine-Tuning, Transformers, Python, Deep Learning


