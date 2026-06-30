import sys
import os
import torch
import torch.nn.functional as F
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

# Shape attendue par uniGradICON pré-entraîné
UNIGRADICON_SHAPE = (175, 175, 175)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# =============================================================================
# RESIZE — uniGradICON attend une shape fixe (175,175,175)
# =============================================================================

def resize_to_unigradicon_shape(tensor: torch.Tensor, target_shape=UNIGRADICON_SHAPE) -> torch.Tensor:
    """
    Redimensionne un tensor (B, C, D, H, W) vers la shape attendue par uniGradICON.
    Utilise l'interpolation trilinéaire (adaptée aux volumes 3D continus comme CT/PET).

    ⚠️ Perte de résolution spatiale : CT natif (389,512,512) → (175,175,175).
    Point à valider avec la prof — alternative possible: patch-based training.
    """
    assert tensor.dim() == 5, f"Tensor doit être (B,C,D,H,W), reçu shape {tensor.shape}"
    return F.interpolate(
        tensor,
        size=target_shape,
        mode='trilinear',
        align_corners=False
    )


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
# TRAINING — version actuelle : fine-tuning sur CT uniquement
# =============================================================================

def train_one_epoch(net, optimizer, loader, epoch):
    net.train()
    losses = []

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        # uniGradICON prend (moving, fixed) — on utilise CT pour la registration
        ct_source = batch['ct_source'].to(device)  # (B, 1, D, H, W)
        ct_target = batch['ct_target'].to(device)  # (B, 1, D, H, W)

        # Resize vers la shape attendue par uniGradICON (175,175,175)
        ct_source = resize_to_unigradicon_shape(ct_source)
        ct_target = resize_to_unigradicon_shape(ct_target)

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

            ct_source = resize_to_unigradicon_shape(ct_source)
            ct_target = resize_to_unigradicon_shape(ct_target)

            loss_object = net(ct_source, ct_target)
            loss = torch.mean(loss_object.all_loss)
            losses.append(loss.item())

    mean_loss = np.mean(losses)
    print(f"  Val   loss: {mean_loss:.4f}")
    return mean_loss


# =============================================================================
# ⚠️ FINE-TUNING CT + PET (futur) — la prof a demandé d'itérer aussi sur PET
# =============================================================================
#
# Idée : entraîner le réseau sur les DEUX modalités, pas seulement CT.
# Deux approches possibles à discuter avec la prof :
#
# APPROCHE A — Loss combinée (CT et PET dans le même forward, pondérée)
# -----------------------------------------------------------------------
# def train_one_epoch_ct_pet(net, optimizer, loader, epoch, pet_weight=0.3):
#     net.train()
#     losses = []
#
#     for batch in tqdm(loader, desc=f"Epoch {epoch}"):
#         ct_source  = resize_to_unigradicon_shape(batch['ct_source'].to(device))
#         ct_target  = resize_to_unigradicon_shape(batch['ct_target'].to(device))
#         pet_source = resize_to_unigradicon_shape(batch['pet_source'].to(device))
#         pet_target = resize_to_unigradicon_shape(batch['pet_target'].to(device))
#
#         optimizer.zero_grad()
#
#         # Registration sur CT (phi calculé sur l'anatomie)
#         loss_object_ct = net(ct_source, ct_target)
#         loss_ct = torch.mean(loss_object_ct.all_loss)
#
#         # Registration sur PET (même réseau, autre paire)
#         loss_object_pet = net(pet_source, pet_target)
#         loss_pet = torch.mean(loss_object_pet.all_loss)
#
#         # Loss combinée pondérée — CT prioritaire (meilleur contraste anatomique)
#         loss = loss_ct + pet_weight * loss_pet
#         loss.backward()
#         optimizer.step()
#
#         losses.append(loss.item())
#
#     mean_loss = np.mean(losses)
#     print(f"  Train loss (CT+PET): {mean_loss:.4f}")
#     return mean_loss
#
#
# APPROCHE B — Alterner les batches CT et PET (curriculum simple)
# -----------------------------------------------------------------------
# def train_one_epoch_alternating(net, optimizer, loader, epoch):
#     net.train()
#     losses = []
#
#     for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
#         if i % 2 == 0:
#             source = resize_to_unigradicon_shape(batch['ct_source'].to(device))
#             target = resize_to_unigradicon_shape(batch['ct_target'].to(device))
#         else:
#             source = resize_to_unigradicon_shape(batch['pet_source'].to(device))
#             target = resize_to_unigradicon_shape(batch['pet_target'].to(device))
#
#         optimizer.zero_grad()
#         loss_object = net(source, target)
#         loss = torch.mean(loss_object.all_loss)
#         loss.backward()
#         optimizer.step()
#
#         losses.append(loss.item())
#
#     mean_loss = np.mean(losses)
#     print(f"  Train loss (alternating CT/PET): {mean_loss:.4f}")
#     return mean_loss
#
# ⚠️ Point à valider avec la prof avant d'implémenter :
#    - pet_weight : quelle pondération entre CT et PET dans la loss ?
#    - Le PET a un bruit/contraste très différent du CT — la même loss
#      similarity (NCC) est-elle adaptée, ou faut-il une loss spécifique PET ?
# =============================================================================


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

    # 4. Boucle d'entraînement (CT uniquement pour l'instant)
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