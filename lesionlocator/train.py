import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, '/home/chiara/LesionLocator/lesionlocator')
sys.path.insert(0, '/home/chiara/uniGradICON/src')

from dataset_normalized import CTPETDataset, train_val_split

import unigradicon
import icon_registration as icon
from icon_registration.losses import to_floats

# =============================================================================
# CONFIG
# =============================================================================

CT_DIR  = "/scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr"
PET_DIR = "/scratch/nnUNet_raw/Dataset900_USZMelanoma/imagesTr"

EPOCHS      = 50
LR          = 1e-5       # fine-tuning → lr faible
BATCH_SIZE  = 1          # images 3D volumineuses
SAVE_DIR    = "/home/chiara/checkpoints"
LOG_DIR     = "/home/chiara/logs"
EVAL_EVERY  = 5          # évaluer tous les 5 epochs

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# DATALOADER
# =============================================================================

def get_dataloaders():
    train_ids, val_ids = train_val_split(ct_dir=CT_DIR, pet_dir=PET_DIR, seed=42)

    train_ds = CTPETDataset(
        ct_dir=CT_DIR, pet_dir=PET_DIR,
        patient_ids=train_ids,
        bidirectional=True,
        resample_pet=True
    )
    val_ds = CTPETDataset(
        ct_dir=CT_DIR, pet_dir=PET_DIR,
        patient_ids=val_ids,
        bidirectional=False,
        resample_pet=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} paires | Val: {len(val_ds)} paires")
    return train_loader, val_loader

# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(net, optimizer, loader, epoch):
    net.train()
    losses = []

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        # uniGradICON prend (moving, fixed) — on utilise CT pour la registration
        ct_source = batch['ct_source'].to(device)  # (B, 1, D, H, W)
        ct_target = batch['ct_target'].to(device)  # (B, 1, D, H, W)

        optimizer.zero_grad()
        loss_object = net(ct_source, ct_target)
        loss = torch.mean(loss_object.all_loss)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    print(f"  Train loss: {mean_loss:.4f}")
    return mean_loss


def evaluate(net, loader, epoch):
    net.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            ct_source = batch['ct_source'].to(device)
            ct_target = batch['ct_target'].to(device)

            loss_object = net(ct_source, ct_target)
            loss = torch.mean(loss_object.all_loss)
            losses.append(loss.item())

    mean_loss = np.mean(losses)
    print(f"  Val   loss: {mean_loss:.4f}")
    return mean_loss


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # 1. Charger le modèle pré-entraîné
    print("Loading pretrained uniGradICON...")
    net = unigradicon.get_unigradicon()
    net = net.to(device)
    print("Model loaded ✓")

    # 2. Optimizer — lr faible pour fine-tuning
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # 3. Dataloaders
    train_loader, val_loader = get_dataloaders()

    # 4. Boucle d'entraînement
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

        train_loss = train_one_epoch(net, optimizer, train_loader, epoch)

        if epoch % EVAL_EVERY == 0:
            val_loss = evaluate(net, val_loader, epoch)

            # Sauvegarder le meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    net.state_dict(),
                    os.path.join(SAVE_DIR, f"unigradicon_finetuned_best.pth")
                )
                print(f"  ✅ Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")

        # Checkpoint régulier
        if epoch % 10 == 0:
            torch.save(
                net.state_dict(),
                os.path.join(SAVE_DIR, f"unigradicon_epoch_{epoch}.pth")
            )

    print("\n✅ Fine-tuning terminé !")
    print(f"   Meilleur val_loss: {best_val_loss:.4f}")
    print(f"   Modèle sauvegardé: {SAVE_DIR}/unigradicon_finetuned_best.pth")