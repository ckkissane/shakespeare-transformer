import torch
import torch.nn.functional as F
from model.sample_fns import sample
from model.transformer import DecoderOnlyTransformer
from dataset.word_dataset import WordDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import re


def train_shakespeare(model, train_dataset, batch_size=128, lr=6e-4, max_epochs=10):
    """Standard PyTorch training loop"""

    print(f"batch_size: {batch_size}")
    print(f"lr: {lr}")
    print(f"max_epochs: {max_epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader = DataLoader(
        train_dataset, shuffle=True, pin_memory=True, batch_size=batch_size
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for it, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            optimizer.step()

            pbar.set_description(
                f"epoch {epoch} iter {it}: train loss {loss.item():.5f}"
            )

        # Eval
        model.eval()
        with torch.no_grad():
            context = "\nO God, O God!"
            x = torch.tensor(
                [train_dataset.stoi[s] for s in re.split(r"\b", context)],
                dtype=torch.long,
            )[None, ...].to(device)
            y = sample(model, x, 400, temperature=1.0, sample=True, top_k=10)[0]
            completion = "".join([train_dataset.itos[int(i)] for i in y])
            print("sample:", completion[: completion.rfind("\n")])

        # Save model
        print("saving model")
        ckpt_path = os.path.join(os.getcwd(), "model.pt")
        torch.save(model.state_dict(), ckpt_path)
        model.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_epochs", type=int, default=15)
    args = parser.parse_args()

    shakespeare_corpus_path = "./dataset/100-0.txt"
    text = open(shakespeare_corpus_path, "r", encoding="utf-8-sig").read()
    train_dataset = WordDataset(text, block_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = (
        DecoderOnlyTransformer(
            num_layers=8,
            num_heads=8,
            vocab_size=train_dataset.vocab_size,
            hidden_size=512,
            max_pos_embeddings=train_dataset.block_size,
            dropout=0.1,
        )
        .to(device)
        .train()
    )
    train_shakespeare(
        model,
        train_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
    )
