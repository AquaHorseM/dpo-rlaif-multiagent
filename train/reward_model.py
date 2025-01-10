import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from tqdm import tqdm

class RewardDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        """
        Dataset for reward model training.

        Args:
            data_file (str): Path to the JSON dataset file.
            tokenizer (AutoTokenizer): Tokenizer for encoding text inputs.
            max_length (int): Maximum token length.
        """
        self.data = self.load_data(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def load_data(data_file):
        with open(data_file, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = example["prompt"]
        pos_answer = example["pos_answers"][0]
        neg_answer = example["neg_answers"][0]

        pos_input = f"{prompt}\n\n{pos_answer}"
        neg_input = f"{prompt}\n\n{neg_answer}"

        pos_encoding = self.tokenizer(
            pos_input,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        neg_encoding = self.tokenizer(
            neg_input,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": torch.cat([pos_encoding["input_ids"], neg_encoding["input_ids"]]),
            "attention_mask": torch.cat([pos_encoding["attention_mask"], neg_encoding["attention_mask"]]),
            "labels": torch.tensor([1.0, 0.0]),  # Positive = 1.0, Negative = 0.0
        }


def train_reward_model(data_file, model_name, output_dir, num_epochs=3, batch_size=8, lr=5e-5, max_length=512):
    """
    Train a reward model using the provided dataset.

    Args:
        data_file (str): Path to the JSON dataset file.
        model_name (str): Pretrained model name or path.
        output_dir (str): Directory to save the trained model.
        num_epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        lr (float): Learning rate.
        max_length (int): Maximum token length.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # Single regression score

    # Load dataset
    dataset = RewardDataset(data_file, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    model.train()
    progress_bar = tqdm(range(num_training_steps), desc="Training")
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"epoch": epoch + 1, "loss": loss.item()})

    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a reward model using a preference dataset.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the JSON dataset file.")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name or path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length.")

    args = parser.parse_args()
    train_reward_model(
        data_file=args.data_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )
