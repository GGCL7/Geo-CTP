import torch
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, confusion_matrix
import numpy as np
import os

# Configuration
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
MAX_LENGTH = 100
BATCH_SIZE = 8  # Consider reducing if OOM errors occur
EPOCHS = 5
LEARNING_RATE = 2e-5
RANDOM_SEED = 42
BEST_MODEL_DIR = "bestmodel" # Directory to save the best model and tokenizer

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Function to read labels from FASTA headers
def read_labels(fasta_path):
    """
    Reads labels from a FASTA file.
    Label is 1 if header starts with '>DCTPep', 0 otherwise.
    """
    labels = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            if line.strip().startswith('>DCTPep'):
                labels.append(1)
            else:
                labels.append(0)
    return labels

# Function to read sequences and corresponding labels from a FASTA file
def read_fasta_sequences_and_labels(fasta_path):
    """
    Reads sequences and their labels from a FASTA file.
    """
    sequences = []
    current_labels = read_labels(fasta_path)
    current_sequence = ""
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = ""
            else:
                current_sequence += line
        if current_sequence: # Add the last sequence
            sequences.append(current_sequence)

    if len(sequences) != len(current_labels):
        raise ValueError(f"Mismatch between number of sequences ({len(sequences)}) and labels ({len(current_labels)}) in {fasta_path}")
    return sequences, current_labels

# Custom PyTorch Dataset
class PeptideDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = int(self.labels[idx])

        # Tokenize the sequence
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True, # Add [CLS] and [SEP] tokens
            max_length=self.max_length,
            return_token_type_ids=False, # Not needed for ESM classification
            padding='max_length',    # Pad/truncate to max_length
            truncation=True,
            return_attention_mask=True, # Return attention mask
            return_tensors='pt',     # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to calculate metrics
def calculate_metrics(labels, preds):
    """
    Calculates ACC, SEN, SPE, MCC from true labels and predictions.
    """
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    # Ensure cm is 2x2 even if predictions are all one class, or labels are all one class
    # This is important for correctly unpacking tn, fp, fn, tp
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    # Sensitivity (Recall for positive class)
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity (Recall for negative class)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return acc, sen, spe, mcc

def main():
    # Check if data files exist
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: Training file '{TRAIN_FILE}' not found.")
        return
    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file '{TEST_FILE}' not found.")
        return

    # Create directory for saving the best model
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading model: {MODEL_NAME}")
    model = EsmForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # Binary classification

    # Load data
    print(f"Reading training data from {TRAIN_FILE}...")
    train_sequences, train_labels = read_fasta_sequences_and_labels(TRAIN_FILE)
    print(f"Reading test data from {TEST_FILE}...")
    test_sequences, test_labels = read_fasta_sequences_and_labels(TEST_FILE)

    print(f"Number of training samples: {len(train_sequences)}")
    print(f"Number of test samples: {len(test_sequences)}")

    # Create datasets and dataloaders
    train_dataset = PeptideDataset(train_sequences, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = PeptideDataset(test_sequences, test_labels, tokenizer, MAX_LENGTH)

    # Using num_workers > 0 can speed up data loading if your system supports it well.
    # On some systems (like certain cloud notebooks), it might cause issues. Set to 0 if you encounter problems.
    num_data_workers = 2 if torch.cuda.is_available() else 0 # Use 0 for CPU to avoid potential issues
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_data_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_data_workers)


    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    best_test_accuracy = 0.0 # Keep track of the best test accuracy

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        total_train_loss = 0
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        for batch_num, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()

            if (batch_num + 1) % 50 == 0: # Print progress every 50 batches
                print(f"  Batch {batch_num + 1}/{len(train_dataloader)}, Training Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average Training Loss for Epoch {epoch + 1}: {avg_train_loss:.4f}")

        # --- Per-epoch EVALUATION on Test Set ---
        print(f"\n--- Evaluating on test set for Epoch {epoch + 1} ---")
        model.eval() # Set model to evaluation mode
        all_epoch_preds = []
        all_epoch_labels = []

        with torch.no_grad(): # Disable gradient calculations for evaluation
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_epoch_preds.extend(preds)
                all_epoch_labels.extend(labels_batch.cpu().numpy())

        # Calculate metrics for the current epoch's test evaluation
        if len(all_epoch_labels) > 0 and len(all_epoch_preds) > 0:
            current_acc, current_sen, current_spe, current_mcc = calculate_metrics(np.array(all_epoch_labels), np.array(all_epoch_preds))
            print(f"Epoch {epoch+1} Test Metrics: ACC: {current_acc:.4f}, SEN: {current_sen:.4f}, SPE: {current_spe:.4f}, MCC: {current_mcc:.4f}")

            # Check if this is the best model so far based on test accuracy
            if current_acc > best_test_accuracy:
                best_test_accuracy = current_acc
                print(f"New best test accuracy: {best_test_accuracy:.4f}. Saving model and tokenizer to '{BEST_MODEL_DIR}' directory...")
                model.save_pretrained(BEST_MODEL_DIR)
                tokenizer.save_pretrained(BEST_MODEL_DIR) # Save tokenizer alongside the model
        else:
            print(f"Epoch {epoch+1} Test Metrics: Not enough data to calculate metrics (labels or predictions might be empty).")
        # --- End Per-epoch EVALUATION ---

    print("\nTraining complete.")
    print(f"Best test accuracy achieved during training: {best_test_accuracy:.4f}")
    print(f"Best model and tokenizer saved to: {BEST_MODEL_DIR}")

if __name__ == "__main__":
    main()