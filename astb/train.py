"""Training loop for TinyGPT on the ASTB synthetic datasets."""

import random
import torch
import torch.nn.functional as F
from . import vocab as V
from .data import to_sequence


def make_batches(examples, batch_size, rng, shuffle=True):
    idxs = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield [examples[j] for j in idxs[i:i + batch_size]]


def collate(batch):
    seqs, starts = [], []
    for ex in batch:
        seq, start = to_sequence(ex)
        seqs.append(seq)
        starts.append(start)
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), V.PAD_ID, dtype=torch.long)
    loss_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, (seq, start) in enumerate(zip(seqs, starts)):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        # predict tokens from (start-1) -> output tokens + eos, i.e. targets
        # at position t are input_ids[t+1]; mark positions t in [start-1, len-2]
        loss_mask[i, start - 1: len(seq) - 1] = True
    return input_ids, loss_mask


def train_model(model, examples, epochs, batch_size, lr, seed, device="cpu", verbose=False):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history = []
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for batch in make_batches(examples, batch_size, rng):
            input_ids, loss_mask = collate(batch)
            input_ids = input_ids.to(device)
            loss_mask = loss_mask.to(device)
            logits = model(input_ids)  # (B, T, V)
            targets = input_ids[:, 1:]
            preds = logits[:, :-1, :]
            mask = loss_mask[:, :-1]
            loss_per_tok = F.cross_entropy(
                preds.reshape(-1, preds.size(-1)), targets.reshape(-1), reduction="none"
            ).view(targets.shape)
            loss = (loss_per_tok * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        history.append(avg_loss)
        if verbose:
            print(f"  epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")
    model.eval()
    return history
