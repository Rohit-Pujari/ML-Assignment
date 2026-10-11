"""Fine-tune one shared DistilBERT encoder with six aspect heads."""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from data import ASPECTS, load_data, polarity

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
LABELS = ["negative", "neutral", "positive"]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReviewDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=MAX_LENGTH):
        self.texts = frame["text"].tolist()
        labels = np.full((len(frame), len(ASPECTS)), -100, dtype=np.int64)
        for col, aspect in enumerate(ASPECTS):
            values = frame[aspect].map(polarity).to_numpy()
            labels[:, col] = np.where(values < 0, -100, values)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        tokens = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length)
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
                "labels": self.labels[i]}


def collate_batch(tokenizer):
    def collate(items):
        encoded = tokenizer.pad(
            [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in items],
            padding=True, return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(np.stack([x["labels"] for x in items]), dtype=torch.long)
        return encoded
    return collate


class MultiAspectDistilBert(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            self.encoder = AutoModel.from_config(config)
        hidden = self.encoder.config.dim
        self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in ASPECTS])

    def forward(self, input_ids, attention_mask):
        # DistilBERT's first token is the [CLS]-equivalent pooled representation.
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        return torch.stack([head(pooled) for head in self.heads], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Train on 2,000 rows for one epoch")
    args = parser.parse_args()
    started = time.perf_counter()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU unavailable; using CPU")

    train, test = load_data()
    if args.smoke:
        train = train.sample(n=2000, random_state=42).reset_index(drop=True)
        epochs = 1
        batch_size = 16
    else:
        epochs = 2
        batch_size = 32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(train, tokenizer)
    test_ds = ReviewDataset(test, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=device.type == "cuda",
        collate_fn=collate_batch(tokenizer), num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, pin_memory=device.type == "cuda",
        collate_fn=collate_batch(tokenizer), num_workers=0,
    )

    class_weights = []
    for aspect in ASPECTS:
        labels = train.loc[train[aspect] != -1, aspect].map(polarity)
        counts = np.bincount(labels, minlength=3)
        weights = len(labels) / (3.0 * np.maximum(counts, 1))
        class_weights.append(weights)
    class_weights = torch.tensor(np.stack(class_weights), dtype=torch.float32, device=device)
    model = MultiAspectDistilBert().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not bf16)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    model.train()
    global_step = 0
    for epoch in range(epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            micro_batches = [batch]
            try:
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=device.type == "cuda"):
                    logits = model(batch["input_ids"], batch["attention_mask"])
                    losses = []
                    for i in range(len(ASPECTS)):
                        per_item = nn.functional.cross_entropy(
                            logits[:, i, :].float(), batch["labels"][:, i],
                            weight=class_weights[i], ignore_index=-100, reduction="none",
                        )
                        valid = batch["labels"][:, i] != -100
                        losses.append(per_item.sum() / valid.sum().clamp_min(1))
                    loss = torch.stack(losses).sum()
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda" or batch["input_ids"].shape[0] <= 16:
                    raise
                optimizer.zero_grad(set_to_none=True)
                if "logits" in locals():
                    del logits
                torch.cuda.empty_cache()
                micro_batches = [
                    {key: value[start:start + 16] for key, value in batch.items()}
                    for start in range(0, batch["input_ids"].shape[0], 16)
                ]
                print("CUDA OOM: retrying this batch as micro-batches of 16")
                loss_values = []
                for micro in micro_batches:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(micro["input_ids"], micro["attention_mask"])
                        per_aspect = []
                        for i in range(len(ASPECTS)):
                            per_item = nn.functional.cross_entropy(
                                logits[:, i, :].float(), micro["labels"][:, i],
                                weight=class_weights[i], ignore_index=-100, reduction="none",
                            )
                            valid = micro["labels"][:, i] != -100
                            per_aspect.append(per_item.sum() / valid.sum().clamp_min(1))
                        micro_loss = torch.stack(per_aspect).sum() / len(micro_batches)
                    scaler.scale(micro_loss).backward()
                    loss_values.append(micro_loss.detach())
                loss = torch.stack(loss_values).sum()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if global_step % 200 == 0:
                print(f"step {global_step}: loss={loss.item():.4f}")

    model.eval()
    all_logits = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                all_logits.append(model(batch["input_ids"], batch["attention_mask"]).float().cpu())
    pred = torch.cat(all_logits).argmax(dim=-1).numpy()
    records = []
    for i, aspect in enumerate(ASPECTS):
        truth = test[aspect].map(polarity).to_numpy()
        rated = truth >= 0
        records.append({
            "aspect": aspect, "model": "DistilBERT",
            "accuracy": accuracy_score(truth[rated], pred[rated, i]),
            "macro_f1": f1_score(truth[rated], pred[rated, i], average="macro"),
        })
    root = Path(__file__).resolve().parent
    (root / "models" / "bert").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), root / "models" / "bert" / "model.pt")
    model.encoder.config.save_pretrained(root / "models" / "bert")
    tokenizer.save_pretrained(root / "models" / "bert")
    results = pd.DataFrame(records)
    results.to_csv(root / "results" / "bert_results.csv", index=False)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    else:
        print("Peak GPU memory: unavailable (CPU run)")
    print(f"Elapsed: {time.perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    main()
