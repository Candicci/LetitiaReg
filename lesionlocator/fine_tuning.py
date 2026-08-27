import sys
import os
import time
import argparse
import logging
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


EPOCHS      = 100        # plafond large — on garde le meilleur modèle sur val_loss
LR          = 1e-5       # fine-tuning → lr faible
BATCH_SIZE  = 1          # images 3D volumineuses
SAVE_DIR    = "/home/chiara/checkpoints"
LOG_DIR     = "/home/chiara/logs"
EVAL_EVERY  = 1          # évaluer à chaque epoch (run potentiellement coupé)

# Shape attendue par uniGradICON pré-entraîné
UNIGRADICON_SHAPE = (175, 175, 175)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# LOGGING — (B) écrire dans un fichier ET dans le terminal
# =============================================================================

def setup_logging():
    """
    Configure le logging pour écrire à la fois dans la console et dans un fichier.
    Le fichier permet de récupérer la courbe de loss après coup, même si le
    terminal est fermé ou le pod redémarré.
    """
    log_file = os.path.join(LOG_DIR, f"train_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging vers {log_file}")
    return log_file


# =============================================================================
# RESIZE — uniGradICON attend une shape fixe (175,175,175)
# =============================================================================

def resize_to_unigradicon_shape(tensor: torch.Tensor, target_shape=UNIGRADICON_SHAPE) -> torch.Tensor:
    """
    Redimensionne un tensor (B, C, D, H, W) vers la shape attendue par uniGradICON.
    Interpolation trilinéaire (adaptée aux volumes 3D continus CT/PET).

    ⚠️ Perte de résolution : CT natif (389,512,512) → (175,175,175).
    Point à valider avec la prof — alternative: patch-based training.
    """
    assert tensor.dim() == 5, f"Tensor doit être (B,C,D,H,W), reçu {tensor.shape}"
    return F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)


# =============================================================================
# DATALOADER
# =============================================================================

def get_dataloaders():
    train_ids, val_ids = train_val_split(ct_dir=CT_DIR, pet_dir=PET_DIR, seed=42)

    train_ds = CTPETDataset(
        ct_dir=CT_DIR, pet_dir=PET_DIR,
        patient_ids=train_ids, bidirectional=True, resample_pet=True
    )
    val_ds = CTPETDataset(
        ct_dir=CT_DIR, pet_dir=PET_DIR,
        patient_ids=val_ids, bidirectional=False, resample_pet=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logging.info(f"Train: {len(train_ds)} paires | Val: {len(val_ds)} paires")
    return train_loader, val_loader


# =============================================================================
# TRAINING — fine-tuning sur CT, avec (A) protection OOM GPU
# =============================================================================

def train_one_epoch(net, optimizer, loader, epoch):
    net.train()
    losses = []
    skipped = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        try:
            ct_source = resize_to_unigradicon_shape(batch['ct_source'].to(device))
            ct_target = resize_to_unigradicon_shape(batch['ct_target'].to(device))

            optimizer.zero_grad()
            loss_object = net(ct_source, ct_target)
            loss = torch.mean(loss_object.all_loss)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        except RuntimeError as e:
            # (A) Si OOM GPU sur une paire : on skip au lieu de tout perdre
            if "out of memory" in str(e).lower():
                skipped += 1
                logging.warning(f"  OOM sur une paire (patient {batch['patient_id']}), skip. "
                                f"Total skipped: {skipped}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            else:
                raise  # autre erreur → on la laisse remonter

        finally:
            # Libérer la mémoire GPU après chaque paire
            if 'ct_source' in locals():
                del ct_source
            if 'ct_target' in locals():
                del ct_target

    mean_loss = float(np.mean(losses)) if losses else float('nan')
    logging.info(f"  Train loss: {mean_loss:.4f} ({len(losses)} paires, {skipped} skipped)")
    return mean_loss


def evaluate(net, loader, epoch):
    net.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            try:
                ct_source = resize_to_unigradicon_shape(batch['ct_source'].to(device))
                ct_target = resize_to_unigradicon_shape(batch['ct_target'].to(device))

                loss_object = net(ct_source, ct_target)
                loss = torch.mean(loss_object.all_loss)
                losses.append(loss.item())

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            finally:
                if 'ct_source' in locals():
                    del ct_source
                if 'ct_target' in locals():
                    del ct_target

    mean_loss = float(np.mean(losses)) if losses else float('nan')
    logging.info(f"  Val   loss: {mean_loss:.4f} ({len(losses)} paires)")
    return mean_loss


# =============================================================================
# CHECKPOINTS — (C) reprise après crash + (D) sauvegarde à chaque epoch
# =============================================================================

def save_checkpoint(net, optimizer, epoch, best_val_loss, path):
    """
    Sauvegarde l'état complet (modèle + optimizer + epoch + best_val_loss)
    pour permettre une reprise exacte après crash.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)


def load_checkpoint(net, optimizer, path):
    """
    Recharge un checkpoint et retourne (epoch_de_depart, best_val_loss).
    """
    logging.info(f"Reprise depuis {path}")
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    logging.info(f"  Reprise à l'epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


# =============================================================================
# ⚠️ FINE-TUNING CT + PET (futur) — la prof a demandé d'itérer aussi sur PET
# =============================================================================
#
# APPROCHE A — Loss combinée pondérée
# -----------------------------------------------------------------------
# def train_one_epoch_ct_pet(net, optimizer, loader, epoch, pet_weight=0.3):
#     for batch in tqdm(loader, desc=f"Epoch {epoch}"):
#         ct_source  = resize_to_unigradicon_shape(batch['ct_source'].to(device))
#         ct_target  = resize_to_unigradicon_shape(batch['ct_target'].to(device))
#         pet_source = resize_to_unigradicon_shape(batch['pet_source'].to(device))
#         pet_target = resize_to_unigradicon_shape(batch['pet_target'].to(device))
#         optimizer.zero_grad()
#         loss_ct  = torch.mean(net(ct_source,  ct_target).all_loss)
#         loss_pet = torch.mean(net(pet_source, pet_target).all_loss)
#         loss = loss_ct + pet_weight * loss_pet
#         loss.backward()
#         optimizer.step()
#
# APPROCHE B — Alterner les batches CT et PET
# -----------------------------------------------------------------------
# def train_one_epoch_alternating(net, optimizer, loader, epoch):
#     for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
#         if i % 2 == 0:
#             source = resize_to_unigradicon_shape(batch['ct_source'].to(device))
#             target = resize_to_unigradicon_shape(batch['ct_target'].to(device))
#         else:
#             source = resize_to_unigradicon_shape(batch['pet_source'].to(device))
#             target = resize_to_unigradicon_shape(batch['pet_target'].to(device))
#         optimizer.zero_grad()
#         loss = torch.mean(net(source, target).all_loss)
#         loss.backward()
#         optimizer.step()
#
# ⚠️ À valider avec la prof :
#    - pet_weight : quelle pondération CT vs PET ?
#    - NCC adaptée au PET, ou loss spécifique nécessaire ?
# =============================================================================


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", default="", type=str,
                        help="Chemin d'un checkpoint pour reprendre l'entraînement")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Device: {device}")

    # 1. Modèle pré-entraîné
    logging.info("Loading pretrained uniGradICON...")
    net = unigradicon.get_unigradicon()
    net = net.to(device)
    logging.info("Model loaded")

    # 2. Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # 3. (C) Reprise éventuelle depuis un checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume_from:
        start_epoch, best_val_loss = load_checkpoint(net, optimizer, args.resume_from)

    # 4. Dataloaders
    train_loader, val_loader = get_dataloaders()

    # 5. Boucle d'entraînement
    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        logging.info(f"--- Epoch {epoch}/{EPOCHS} ---")

        train_loss = train_one_epoch(net, optimizer, train_loader, epoch)

        # (D) Sauvegarde du "last" à CHAQUE epoch (reprise après crash)
        save_checkpoint(net, optimizer, epoch, best_val_loss,
                        os.path.join(SAVE_DIR, "unigradicon_last.pth"))

        # Évaluation
        if epoch % EVAL_EVERY == 0:
            val_loss = evaluate(net, val_loader, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(net, optimizer, epoch, best_val_loss,
                                os.path.join(SAVE_DIR, "unigradicon_best.pth"))
                logging.info(f"  Nouveau meilleur modele (val_loss={val_loss:.4f})")

        dt = time.time() - t0
        logging.info(f"  Epoch {epoch} terminee en {dt/60:.1f} min")

    logging.info("Fine-tuning termine")
    logging.info(f"  Meilleur val_loss: {best_val_loss:.4f}")
    logging.info(f"  Modeles dans: {SAVE_DIR}")
