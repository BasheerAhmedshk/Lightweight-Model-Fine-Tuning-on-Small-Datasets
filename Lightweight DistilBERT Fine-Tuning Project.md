# Lightweight DistilBERT Fine-Tuning Project

This project demonstrates how to fine-tune the `distilbert-base-uncased` model for sequence classification on a small dataset (SST-2 subset) using lightweight techniques suitable for a standard CPU laptop/desktop.

It includes scripts for:
1.  **Data Preparation (`prepare_dataset.py`):** Downloads the SST-2 dataset, selects a small subset (1000 training, 200 validation samples), tokenizes the text, and saves the processed data.
2.  **Model Training (`train_model.py`):** Loads the pre-trained DistilBERT model, applies LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning (can be switched to standard classification head fine-tuning), and trains the model on the prepared dataset using the CPU.

## Setup Instructions

1.  **Clone/Download Project:** Get the project files (`prepare_dataset.py`, `train_model.py`, `requirements.txt`, `README.md`).

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    *   **PyTorch (CPU Version):** First, install the CPU version of PyTorch. The exact command might vary based on your OS. Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select Stable, your OS, Pip, Python, and CPU.
        *Example command (check the PyTorch website for the latest):*
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```
    *   **Other Libraries:** Install the remaining libraries using the provided `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

## Running the Project

1.  **Prepare the Dataset:**
    Run the data preparation script. This will download the SST-2 dataset (if not already cached) and create a `processed_data` directory containing the tokenized training and validation sets.
    ```bash
    python prepare_dataset.py
    ```

2.  **Run Model Training:**
    Execute the training script. This will load the model, configure LoRA (by default), train for one epoch on the CPU, evaluate the model, and save the results (including the trained model/adapter) into the `training_output` directory.
    ```bash
    python train_model.py
    ```
    *   **Note:** Training on a CPU will be significantly slower than on a GPU. The script is configured for a short run (1 epoch) for demonstration purposes.
    *   **Tuning Method:** You can switch between LoRA and standard classification head fine-tuning by changing the `USE_LORA` variable at the top of `train_model.py` (`True` for LoRA, `False` for standard fine-tuning).

## Project Files

*   `prepare_dataset.py`: Script to download and preprocess the dataset.
*   `train_model.py`: Script to configure, train, and evaluate the model.
*   `requirements.txt`: List of Python libraries required.
*   `README.md`: This file.
*   `processed_data/`: Directory created by `prepare_dataset.py` containing tokenized data.
*   `training_output/`: Directory created by `train_model.py` containing logs, checkpoints, and the final trained model/adapter.
*   `todo.md`: Checklist used during development.

## Notes

*   The model achieves around 55% accuracy after one epoch of LoRA tuning on this small subset. More training epochs or a larger dataset would be needed for better performance.
*   The primary goal was to set up a lightweight training pipeline runnable on a CPU, demonstrating LoRA configuration.

