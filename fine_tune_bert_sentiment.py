import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


# Load your labeled dataset
df = pd.read_csv('reviewslabeled.csv')

# Map textual labels to integers
label_map = {'negative': 0, 'positive': 1}
df['label'] = df['label'].map(label_map)

# Debug info
print(df['label'].value_counts())
print("Total rows in CSV:", len(df))
print("Rows with valid labels:", df['label'].notna().sum())
print("Rows with NaN labels:")
print(df[df['label'].isna()])

# Drop invalid rows
df = df.dropna(subset=['label'])

# Split data
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'])

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['review_text'], padding="max_length", truncation=True, max_length=128)

# Tokenize
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Evaluation metric
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",  # Change to "tensorboard" if you want to use TensorBoard
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Final evaluation
eval_results = trainer.evaluate()
print("Final Evaluation Results:", eval_results)

# Save model and tokenizer
trainer.save_model('./sentiment_bert_model')
tokenizer.save_pretrained('./sentiment_bert_model')

# Load model and tokenizer for inference
model = BertForSequenceClassification.from_pretrained('./sentiment_bert_model')
tokenizer = BertTokenizer.from_pretrained('./sentiment_bert_model')
model.to(device)
model.eval()

# Label decoding
label_map_inv = {0: "negative", 1: "positive"}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    return predicted_class_id

# Load your new dataset (CSV with column: 'review_text')
new_df = pd.read_csv('new_reviews.csv')

# Tokenize the new reviews in the same way as the training set
new_dataset = Dataset.from_pandas(new_df)
new_dataset = new_dataset.map(tokenize_function, batched=True)  # Use your original tokenize function
new_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Now, make predictions on the new reviews
predictions = [predict(review) for review in new_df['review_text']]

# Map predictions to sentiment labels
label_map_inv = {0: "negative", 1: "positive",}
predicted_labels = [label_map_inv[pred] for pred in predictions]

# Add predictions to DataFrame
new_df['predicted_label'] = predicted_labels

# Save the results to a new CSV
new_df.to_csv('predicted_reviews.csv', index=False)

print("Predictions have been saved to 'predicted_reviews.csv'")

# Example predictions
print(predict("This hotel was excellent!"))  # Expect: 2 (positive)
print(label_map_inv[predict("The room was dirty and the staff was rude.")])  # Expect: "negative"

# Load the labeled dataset again (ground truth)
df = pd.read_csv('reviewslabeled.csv')
label_map = {'negative': 0, 'positive': 1}
df['label'] = df['label'].map(label_map)

# Load predictions file (output from your model)
predicted_df = pd.read_csv('predicted_reviews.csv')

# Make sure both DataFrames are aligned and contain same number of rows
assert len(df) == len(predicted_df), "Mismatch in number of rows!"

# Map predicted labels to numeric values
label_map = {'negative': 0, 'positive': 1}
predicted_df['predicted_label'] = predicted_df['predicted_label'].map(label_map)

# Compute accuracy
accuracy = accuracy_score(df['label'], predicted_df['predicted_label'])
print(f"Accuracy: {accuracy * 100:.2f}%")

# Optional: Print detailed report
print("\nClassification Report:")
print(classification_report(df['label'], predicted_df['predicted_label'], target_names=['negative', 'positive']))

# Your classification report
report = classification_report(df['label'], predicted_df['predicted_label'], target_names=['negative', 'positive'], output_dict=True)

# Extract values from the classification report for visualization
labels = ['negative', 'positive']
precision = [report[label]['precision'] for label in labels]
recall = [report[label]['recall'] for label in labels]
f1_score = [report[label]['f1-score'] for label in labels]

# Set up the bar chart
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))

rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1-score by Sentiment')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Function to add labels on top of bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add labels
add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# Show the plot
plt.show()