import time
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from src.data import get_dataloaders
from src.utils import DEVICE

_, _, test_dl = get_dataloaders(batch_size=256)

@torch.no_grad()
def recon_metrics(recon_fn):
    """recon_fn: takes x (B,1,28,28) in [0,1], returns x_hat of the same shape in [0,1]."""
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    mse_sum, psnr_sum, n = 0, 0, 0
    for x, _ in test_dl:
        x = x.to(DEVICE)
        x_hat = recon_fn(x).view_as(x).clamp(0, 1)
        mse_per_img = ((x_hat - x) ** 2).flatten(1).mean(1) # (B,)
        mse_sum  += mse_per_img.sum().item()
        psnr_sum += (10 * torch.log10(1.0 / mse_per_img)).sum().item()
        ssim.update(x_hat, x)
        n += x.size(0)
    return {"mse": mse_sum / n, "psnr": psnr_sum / n, "ssim": ssim.compute().item()}

_fid = None  # FID metric that keeps the real (test set) Inception features between calls

@torch.no_grad()
def fid_score(sample_fn, n_fake=5000, batch=100, chunk=100):
    """sample_fn(b) must return b images, shape (b,1,28,28), values in [0,1].
    Real set = the 7,000 test images (their Inception features are computed once and reused).
    Images go through Inception in chunks of `chunk` so a 4 GB GPU does not run out of memory."""
    global _fid
    if _fid is None:
        _fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(DEVICE)
        for x, _ in test_dl:
            for part in x.split(chunk):
                _fid.update(part.to(DEVICE).repeat(1, 3, 1, 1), real=True)
    _fid.reset()  # clears only the fake features (reset_real_features=False)
    made = 0
    while made < n_fake:
        b = min(batch, n_fake - made)
        imgs = sample_fn(b).clamp(0, 1)
        for part in imgs.split(chunk):
            _fid.update(part.repeat(1, 3, 1, 1), real=False)
        made += b
    return _fid.compute().item()

@torch.no_grad()
def latency_ms(fn, reps=20, warmup=3):
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000
