"""
evaluate_ncc.py  — ÉTAPE A/B : test de l'hypothèse avec la métrique NCC uniquement
==================================================================================
Teste si warp(PET_T0, phi) s'aligne mieux sur PET_T1 que PET_T0 seul,
où phi est le champ de déformation calculé par uniGradICON sur (CT_T0, CT_T1).

Pas de masques / pas de Dice ici — on valide d'abord la mécanique du warp PET.

Pour chaque paire :
    net(CT_T0, CT_T1)            -> phi
    warp(PET_T0, phi)           -> PET recalé
    NCC(CT_warped,  CT_T1)      -> contrôle : le recalage CT marche-t-il ?
    NCC(PET_T0,     PET_T1)     -> baseline PET (sans recalage)
    NCC(PET_warped, PET_T1)     -> résultat PET (avec phi du CT)
"""

import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, '/home/chiara/LesionLocator/lesionlocator')
sys.path.insert(0, '/home/chiara/uniGradICON/src')

from dataset_normalized import CTPETDataset, train_val_split, CT_DIR, PET_DIR

import unigradicon
from icon_registration.mermaidlite import compute_warped_image_multiNC

RESULTS_DIR = "/home/chiara/results"
UNIGRADICON_SHAPE = (175, 175, 175)
os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging():
    log_file = os.path.join(RESULTS_DIR, f"eval_ncc_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return log_file


def resize(tensor):
    """Resize (B,C,D,H,W) vers 175^3, interpolation trilinéaire (images continues)."""
    return F.interpolate(tensor, size=UNIGRADICON_SHAPE, mode='trilinear', align_corners=False)


def ncc(a, b, eps=1e-8):
    """Normalized Cross-Correlation dans [-1,1]. 1 = identiques structurellement."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = (a - a.mean()) / (a.std() + eps)
    b = (b - b.mean()) / (b.std() + eps)
    return float((a * b).mean().item())


def evaluate_pair(net, sample):
    patient_id = sample['patient_id']
    tp_source, tp_target = sample['timepoints']

    # (1,1,D,H,W) -> resize 175^3
    ct_source  = resize(sample['ct_source'].unsqueeze(0).to(device))
    ct_target  = resize(sample['ct_target'].unsqueeze(0).to(device))
    pet_source = resize(sample['pet_source'].unsqueeze(0).to(device))
    pet_target = resize(sample['pet_target'].unsqueeze(0).to(device))

    with torch.no_grad():
        net(ct_source, ct_target)          # calcule phi
        phi = net.phi_AB_vectorfield
        spacing = net.spacing

        ct_warped = compute_warped_image_multiNC(
            ct_source, phi, spacing, spline_order=1, zero_boundary=True)
        pet_warped = compute_warped_image_multiNC(
            pet_source, phi, spacing, spline_order=1, zero_boundary=True)

    m = {
        'patient_id': patient_id,
        'pair': f"{tp_source}->{tp_target}",
        'ncc_ct_control':   ncc(ct_warped, ct_target),
        'ncc_pet_baseline': ncc(pet_source, pet_target),
        'ncc_pet_warped':   ncc(pet_warped, pet_target),
    }
    net.clean()
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--max_pairs", default=-1, type=int)
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Device: {device}")

    logging.info("Loading uniGradICON...")
    net = unigradicon.get_unigradicon()
    if args.checkpoint:
        logging.info(f"Loading fine-tuned weights: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        net.load_state_dict(state)
    net = net.to(device)
    net.eval()

    _, val_ids = train_val_split(ct_dir=CT_DIR, pet_dir=PET_DIR, seed=42)
    val_ds = CTPETDataset(patient_ids=val_ids, bidirectional=False, resample_pet=True)
    logging.info(f"Évaluation sur {len(val_ds)} paires de validation")

    all_metrics = []
    n = len(val_ds) if args.max_pairs < 0 else min(args.max_pairs, len(val_ds))
    for i in range(n):
        try:
            m = evaluate_pair(net, val_ds[i])
            all_metrics.append(m)
            logging.info(
                f"[{i+1}/{n}] {m['patient_id']} {m['pair']} | "
                f"CT ctrl NCC={m['ncc_ct_control']:.3f} | "
                f"PET NCC {m['ncc_pet_baseline']:.3f} -> {m['ncc_pet_warped']:.3f}"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logging.warning(f"[{i+1}/{n}] OOM, skip")
                continue
            raise

    def mean_std(key):
        vals = [m[key] for m in all_metrics if m[key] is not None]
        return (np.mean(vals), np.std(vals)) if vals else (float('nan'), float('nan'))

    logging.info("=" * 70)
    logging.info("RÉSULTATS NCC (moyenne ± écart-type)")
    ct_m, ct_s = mean_std('ncc_ct_control')
    pb_m, pb_s = mean_std('ncc_pet_baseline')
    pw_m, pw_s = mean_std('ncc_pet_warped')
    logging.info(f"Contrôle recalage CT      : {ct_m:.3f} ± {ct_s:.3f}")
    logging.info(f"PET baseline (sans phi)   : {pb_m:.3f} ± {pb_s:.3f}")
    logging.info(f"PET recalé  (avec phi CT) : {pw_m:.3f} ± {pw_s:.3f}")
    logging.info(f"  -> gain NCC             : {pw_m - pb_m:+.3f}")
    logging.info("=" * 70)
    if pw_m > pb_m:
        logging.info("phi calculé sur CT améliore l'alignement PET (NCC) -> encourageant")
    else:
        logging.info("phi calculé sur CT n'améliore pas le PET (NCC) -> à investiguer")
    logging.info("=" * 70)

    import csv
    csv_path = os.path.join(RESULTS_DIR, f"ncc_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
    logging.info(f"CSV : {csv_path}")