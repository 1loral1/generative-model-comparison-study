"""Training and evaluation loops."""
import time

import torch
from torch import nn

from .utils import CKPT_DIR, DEVICE, save_history

def _sync(device):
    """Wait for the GPU to finish before reading the clock (GPU work is asynchronous)."""
    if device == "cuda":
        torch.cuda.synchronize()

# for mlp
@torch.no_grad()
def evaluate_classifier(model, dl, device=DEVICE):
    """Returns (average loss, accuracy) over a DataLoader."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, n = 0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)  # batch size
    return total_loss / n, correct / n

def train_classifier(name, model, opt, train_dl, val_dl, epochs, device=DEVICE):
    """Train with cross-entropy, keep the best-val-accuracy checkpoint, save the history."""
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    best_val_acc = 0
    ckpt = CKPT_DIR / f"{name}.pt"

    for epoch in range(epochs):
        model.train()
        t0 = time.perf_counter()
        total_loss, correct, n = 0, 0, 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)

            opt.zero_grad()  # empty gradients
            loss.backward()  # backprop
            opt.step()       # update weights

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)

        # once per epoch (outside the batch loop)
        _sync(device)
        history["epoch_time"].append(time.perf_counter() - t0)
        history["train_loss"].append(total_loss / n)
        history["train_acc"].append(correct / n)

        val_loss, val_acc = evaluate_classifier(model, val_dl, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt)

        print(f"Epoch {epoch+1:3d}: train_loss={total_loss/n:.4f} | train_acc={correct/n:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {history['epoch_time'][-1]:.1f}s")

    model.load_state_dict(torch.load(ckpt, map_location=device))  # roll back to the best epoch
    save_history(name, history)
    return history

@torch.no_grad()
def extract_latents(model, dl, device=DEVICE):
    """Run a model that returns (output, z) over a DataLoader; returns (z, labels) on the CPU."""
    model.eval()
    zs, ys = [], []
    for x, y in dl:
        zs.append(model(x.to(device))[1].cpu())
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)


# for AE family, DDPM
@torch.no_grad()
def eval_loss(model, loss_fn, dl, device=DEVICE):
    model.eval()
    tot, n = 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        tot += loss_fn(model, x, y).item() * x.size(0)
        n += x.size(0)
    return tot / n


def fit(name, model, loss_fn, opt, train_dl, val_dl, epochs, device=DEVICE):
    """Generic loop: loss_fn(model, x, y) -> scalar loss. Keeps the best-val-loss checkpoint."""
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best = float("inf")
    ckpt = CKPT_DIR / f"{name}.pt"

    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        tot, n = 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model, x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        _sync(device)
        history["epoch_time"].append(time.perf_counter() - t0)
        history["train_loss"].append(tot / n)

        val = eval_loss(model, loss_fn, val_dl, device)
        history["val_loss"].append(val)
        if val < best:  # keep the best-val checkpoint
            best = val
            torch.save(model.state_dict(), ckpt)
        print(f"{name} ep {ep+1:3d} | train {tot/n:.5f} | val {val:.5f} | {history['epoch_time'][-1]:.1f}s")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    save_history(name, history)
    return history
