import os
from datasets import load_dataset
from transformers import AutoTokenizer

# Constants
DATASET_NAME = "sst2" # Using sst2 from GLUE benchmark as an example
MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_TRAIN_SAMPLES = 1000
NUM_VALID_SAMPLES = 200
MAX_LENGTH = 128 # Max sequence length for tokenizer
OUTPUT_DIR = "/home/ubuntu/processed_data"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print(f"Loading dataset: {DATASET_NAME}...")
dataset = load_dataset("glue", DATASET_NAME)

# Select subsets
print(f"Selecting {NUM_TRAIN_SAMPLES} training samples and {NUM_VALID_SAMPLES} validation samples...")
train_dataset = dataset["train"].shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))
valid_dataset = dataset["validation"].shuffle(seed=42).select(range(NUM_VALID_SAMPLES))

# Load tokenizer
print(f"Loading tokenizer for model: {MODEL_CHECKPOINT}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Preprocessing function
def preprocess_function(examples):
    # Tokenize the texts (sentence is the key in sst2)
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

# Apply preprocessing
print("Applying preprocessing to datasets...")
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True)

# Remove original text column and rename label column
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["sentence", "idx"])
tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_train_dataset.set_format("torch")

tokenized_valid_dataset = tokenized_valid_dataset.remove_columns(["sentence", "idx"])
tokenized_valid_dataset = tokenized_valid_dataset.rename_column("label", "labels")
tokenized_valid_dataset.set_format("torch")

# Save processed datasets
print(f"Saving processed datasets to {OUTPUT_DIR}...")
tokenized_train_dataset.save_to_disk(os.path.join(OUTPUT_DIR, "train"))
tokenized_valid_dataset.save_to_disk(os.path.join(OUTPUT_DIR, "validation"))

print("Dataset preparation complete.")

