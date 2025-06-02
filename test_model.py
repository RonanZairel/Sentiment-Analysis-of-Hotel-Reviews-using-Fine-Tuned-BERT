from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import BertTokenizer


# Set the local path to your model directory
model_dir = r'C:\Users\ADMIN\BERT\sentiment_bert_model'

# Manually load the model and tokenizer from the local directory
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Example model
tokenizer.save_pretrained(r'C:\Users\ADMIN\BERT\sentiment_bert_model')  # Save it to your local directory

# Example review
review = "This hotel was amazing! I had a wonderful experience."

# Tokenize the review
inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Load the model from the local directory
model = BertForSequenceClassification.from_pretrained(model_dir)

# Make the prediction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class (0: Negative, 1: Neutral, 2: Positive)
predicted_class = torch.argmax(logits, dim=1).item()

# Map the predicted class back to the sentiment label
label_map_inv = {0: 'negative', 1: 'neutral', 2: 'positive'}
print(f"Review: {review}")
print(f"Predicted sentiment: {label_map_inv[predicted_class]}")

# Load and save pretrained model and tokenizer
model_name = "bert-base-uncased"

# Load
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Save to your local path
save_path = r"C:\Users\ADMIN\BERT\sentiment_bert_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
