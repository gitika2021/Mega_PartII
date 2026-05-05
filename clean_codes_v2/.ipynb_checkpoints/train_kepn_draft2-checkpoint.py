import os
import argparse
import signal
import sys
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasetv2 import *
from models import *
from utils import *

# 🔴 Global flag for termination
terminate_requested = False

# 🔴 Signal handler
def handle_signal(signum, frame):
    global terminate_requested
    print(f"\n⚠️ Received signal {signum}. Saving checkpoint before exit...")
    terminate_requested = True

# =========================
# DDP setup
# =========================
def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# =========================
# DataLoader
# =========================
def get_loader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    ), sampler

# =========================
# Training
# =========================
def train(rank, world_size, args):

    global terminate_requested

    # 🔴 Register signals
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    setup(rank, world_size, backend)

    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)

    model = HybridConvNet(n=args.n).to(device)

    if use_cuda:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    train_dataset = LightCurveDataset(args.data, 'train', device=device)
    val_dataset = LightCurveDataset(args.data, 'val', device=device)

    train_loader, train_sampler = get_loader(train_dataset, args.batch_size, rank, world_size)
    val_loader, _ = get_loader(val_dataset, args.batch_size, rank, world_size)

    latest_ckpt = args.modelpath + "_latest.pth"
    best_ckpt = args.modelpath + "_best.pth"

    start_epoch = 0
    best_val = float("inf")

    # Resume logic
    should_resume = (
        args.resume == "yes" or
        (args.resume == "auto" and os.path.exists(latest_ckpt))
    )

    if should_resume:
        if rank == 0:
            print(f"🔄 Resuming from {latest_ckpt}")

        ckpt = torch.load(latest_ckpt, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))

    elif rank == 0:
        print("🆕 Starting from scratch")

    # =========================
    # Training loop
    # =========================
    for epoch in range(start_epoch, args.epochs):

        model.train()
        train_sampler.set_epoch(epoch)

        train_loss = 0.0

        for lc_batch, _, real_imgs in train_loader:

            # 🔴 If termination requested → break safely
            if terminate_requested:
                break

            lc_input = lc_batch.squeeze(1).to(device)
            real_imgs = real_imgs.to(device)

            optimizer.zero_grad()
            out = model(lc_input.view(lc_input.shape[0], 1, 120))
            loss = symmetry_aware_bce(real_imgs.squeeze(), out.squeeze())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 🔴 Early exit before validation if killed
        if terminate_requested:
            if rank == 0:
                print("⚠️ Termination requested — saving checkpoint mid-epoch")

                ckpt = {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val
                }
                torch.save(ckpt, latest_ckpt)

            break

        # ===== Validation =====
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for lc_batch, _, real_imgs in val_loader:
                lc_input = lc_batch.squeeze(1).to(device)
                real_imgs = real_imgs.to(device)

                out = model(lc_input.view(lc_input.shape[0], 1, 120))
                loss = symmetry_aware_bce(real_imgs.squeeze(), out.squeeze())

                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        # Save (rank 0 only)
        if rank == 0:

            print(f"[Epoch {epoch+1}] Train: {avg_train:.4f} | Val: {avg_val:.4f}")

            ckpt = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val
            }

            torch.save(ckpt, latest_ckpt)

            if avg_val < best_val:
                best_val = avg_val
                torch.save(ckpt, best_ckpt)

    cleanup()

# =========================
# Main
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--modelpath", type=str, required=True)

    parser.add_argument("--device", type=str, default="auto",
                        choices=["cpu", "cuda", "auto"])

    parser.add_argument("--resume", type=str, default="auto",
                        choices=["auto", "yes", "no"])

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device == "cuda":
        world_size = torch.cuda.device_count()
    else:
        world_size = int(os.environ.get("PBS_NP", 4))

    print(f"Device: {args.device} | World size: {world_size}")

    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
