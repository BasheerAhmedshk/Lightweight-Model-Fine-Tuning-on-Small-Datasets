# train_model.py

import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import evaluate

# --- Configuration ---
MODEL_CHECKPOINT = "distilbert-base-uncased"
PROCESSED_DATA_DIR = "/home/ubuntu/processed_data"
OUTPUT_DIR = "/home/ubuntu/training_output"
NUM_LABELS = 2 # For SST-2 (positive/negative)
BATCH_SIZE = 2 # As requested by user (1 or 2)
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1 # Keep it short for a demo/CPU training
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
USE_LORA = True # Set to False to use classification head fine-tuning instead

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Model and Tokenizer ---
print(f"Loading model: {MODEL_CHECKPOINT}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# --- Configure LoRA (if enabled) ---
if USE_LORA:
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, # Sequence Classification
        r=LORA_RANK,                # LoRA rank
        lora_alpha=LORA_ALPHA,      # LoRA alpha
        lora_dropout=LORA_DROPOUT,  # LoRA dropout
        target_modules=["q_lin", "k_lin"] # Target modules for DistilBERT attention layers
    )
    model = get_peft_model(model, lora_config)
    print("LoRA configured model summary:")
    model.print_trainable_parameters()
else:
    print("Using standard classification head fine-tuning (LoRA disabled).")

# --- Load Processed Datasets ---
print(f"Loading processed datasets from {PROCESSED_DATA_DIR}...")
train_dataset = load_from_disk(os.path.join(PROCESSED_DATA_DIR, "train"))
valid_dataset = load_from_disk(os.path.join(PROCESSED_DATA_DIR, "validation"))

# --- Training Arguments ---
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    # Use argument names confirmed by diagnostic script
    eval_strategy="epoch",  # Corrected from evaluation_strategy
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    use_cpu=True, # Use modern argument for CPU training
    # report_to="none" # Disable wandb/tensorboard reporting if not needed
)

# --- Metrics ---
print("Loading metric using 'evaluate' library...")
metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# --- Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --- Start Training ---
print("Starting training...")
trainer.train()

# --- Evaluate ---
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# --- Save Model ---
print(f"Saving final model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
# If using LoRA, save the adapter separately (optional but good practice)
if USE_LORA:
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))

print("Training and evaluation complete.")

