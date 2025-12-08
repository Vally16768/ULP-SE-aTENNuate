import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from attenuate.model import aTENNuate


class VoiceBankDemandDataset(Dataset):
    """
    Dataset bazat pe un CSV cu coloane:
        noisy,clean
    Fiecare rând: calea către wav zgomotos și curat.
    """
    def __init__(
        self,
        csv_path: str,
        segment_len: int = 16000 * 2,  # 2 sec default
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = segment_len

        self.pairs: List[Tuple[Path, Path]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "noisy" not in reader.fieldnames or "clean" not in reader.fieldnames:
                raise ValueError("CSV must contain columns: noisy, clean")
            for row in reader:
                self.pairs.append((Path(row["noisy"]), Path(row["clean"])))
        if not self.pairs:
            raise ValueError(f"No rows found in {csv_path}")

        print(f"[Dataset] Loaded {len(self.pairs)} pairs from {csv_path}")

    def __len__(self):
        return len(self.pairs)

    def _load_mono(self, path: Path) -> torch.Tensor:
        # Dacă torchaudio/codec îți face probleme, se poate schimba pe soundfile/librosa.
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)  # (T,)

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx]
        noisy = self._load_mono(noisy_path)
        clean = self._load_mono(clean_path)

        # Random crop / pad la segment_len
        T = noisy.shape[-1]
        seg = self.segment_len
        if T >= seg:
            start = torch.randint(0, T - seg + 1, (1,)).item()
            noisy_seg = noisy[start:start + seg]
            clean_seg = clean[start:start + seg]
        else:
            pad = seg - T
            noisy_seg = torch.nn.functional.pad(noisy, (0, pad))
            clean_seg = torch.nn.functional.pad(clean, (0, pad))

        return noisy_seg, clean_seg


def train_epoch(model, loader, optimizer, device, epoch: int):
    model.train()
    loss_fn = nn.L1Loss()
    running_loss = 0.0
    n = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", unit="batch")

    for noisy, clean in progress_bar:
        noisy = noisy.to(device)          # (B, T)
        clean = clean.to(device)

        noisy = noisy.unsqueeze(1)        # (B, 1, T)
        clean = clean.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        out = model(noisy)
        loss = loss_fn(out, clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy.size(0)
        n += noisy.size(0)

        avg_loss = running_loss / max(1, n)
        progress_bar.set_postfix({"L1": f"{avg_loss:.6f}"})

    return running_loss / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True,
                    help="CSV VoiceBank-DEMAND train (coloane: noisy,clean)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--segment-len", type=int, default=16000 * 2)
    ap.add_argument("--checkpoint-out", type=str, default="checkpoints/atennuate_fp32.pt")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader num_workers (0 = fără multiprocessing, mai sigur)")
    # hiperparametri pentru scheduler + early stopping
    ap.add_argument("--lr-factor", type=float, default=0.5,
                    help="Factor de reducere LR pentru ReduceLROnPlateau")
    ap.add_argument("--lr-patience", type=int, default=3,
                    help="Patience (în epoci) pentru ReduceLROnPlateau")
    ap.add_argument("--min-lr", type=float, default=1e-6,
                    help="LR minim pentru ReduceLROnPlateau")
    ap.add_argument("--early-stop-patience", type=int, default=7,
                    help="Patience (în epoci) pentru early stopping")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset + DataLoader
    ds = VoiceBankDemandDataset(args.train_csv, segment_len=args.segment_len)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    print(f"[DataLoader] Batches per epoch: {len(dl)}")

    # Model
    model = aTENNuate().to(device)

    # Optimizer + scheduler + early stopping state
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr,
    )

    best_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = args.early_stop_patience

    ckpt_path = Path(args.checkpoint_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch:03d}/{args.epochs} =====")
        loss = train_epoch(model, dl, optimizer, device, epoch)
        print(f"[Epoch {epoch:03d}] mean train L1={loss:.6f}")

        # scheduler pe baza loss-ului mediu pe epocă
        scheduler.step(loss)

        # early stopping + best checkpoint
        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Improved (best L1={best_loss:.6f}); checkpoint saved -> {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s). "
                  f"Best L1 so far: {best_loss:.6f}")

        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")


if __name__ == "__main__":
    main()
