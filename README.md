# Native Adaptation to German Dataset for Dialogue State Tracking

This repository implements a BERT multilingual model (mBERT) fine-tuned for **Dialogue State Tracking (DST)** on German-language datasets. The workflow includes dataset preparation, tokenization, model training with advanced optimization, evaluation, and model deployment.

## Features
- Fine-tuning of `bert-base-multilingual-cased` for binary classification.
- Dynamic padding and tokenization for memory efficiency.
- Training with mixed-precision and gradient accumulation support.
- Custom evaluation metrics and model checkpointing.

## Installation

### Dependencies
- Python 3.10+
- Required libraries:
  
```bash
  pip install torch transformers scikit-learn
```
## Usage

### Dataset Preparation
1. Load Dataset: Ensure the dataset is stored in german_data.json with dialogue logs under the log-de key. Example structure:
```json
CopyEdit
{
  "dialogue_id": {
    "log-de": [
      {"text": "user utterance", "label": 0},
      ...
    ]
  }
}
```

2. Flatten and Split Data:
```python
# Load and flatten dialogues
with open("german_data.json", "r", encoding="utf-8") as f:
    dialogues_data = json.load(f)
dialogues = [{"text": turn["text"], "label": 0} for dialogue in dialogues_data.values() for turn in dialogue["log-de"]]
train_data, test_data = train_test_split(dialogues, test_size=0.2, random_state=42)
```

3. Tokenization and Dataset Class: Use BertTokenizer with a sequence length limit of 128 tokens:
```python
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
```
4. Define a custom GermanDialogueDataset class to handle tokenization and data loading. <br>
4.1 Model Initialization
```python
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
```

4.2 Training Configuration: Configure training arguments using TrainingArguments:
```python
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # Recommended: 3-5 epochs
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
```
5. Training and Evaluation
```python
# Train the model
trainer.train()

# Evaluate
results = trainer.evaluate()
print("Evaluation Loss:", results["eval_loss"])
```
6. Save Model and Tokenizer
```python
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
```

7. Evaluation Results
- Evaluation Loss: 4.18e-06 (very low, indicating strong performance).
- Throughput: ~1.33 samples/second.
- Note: Results were obtained after only 10% of one epoch. Increasing training epochs (3-5 recommended) may further improve performance.

8. Saved Files: After saving, the directory (./saved_model) includes:
- config.json: Model configuration.
- pytorch_model.bin: Trained weights.
- Tokenizer files (vocab.txt, tokenizer_config.json, etc.).

9. Deployment Opportunities: Load the model for predictions:
```python
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained("./saved_model")
```
Thank You!
