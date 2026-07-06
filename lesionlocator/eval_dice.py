"""
evaluate_dice.py  — ÉTAPE C : test de l'hypothèse via le Dice des lésions
=========================================================================
Teste si le champ phi (calculé par uniGradICON sur CT_T0 -> CT_T1) recale
correctement les LÉSIONS du PET.

C'est la métrique la plus proche de la question clinique de LETITIA :
"une fois phi appliqué, la lésion au temps T0 tombe-t-elle au bon endroit
par rapport à la lésion au temps T1 ?"

Labels : on garde UNIQUEMENT le label 1 (tumeur). Le label 2 est ignoré.

Pour chaque paire :
    net(CT_T0, CT_T1)                      -> phi
    warp(masque_lesion_T0, phi)  [nearest] -> masque recalé
    Dice(masque_T0,     masque_T1)         -> baseline (sans recalage)
    Dice(masque_warped, masque_T1)         -> après recalage par phi
"""

import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from datetime import datetime

sys.path.insert(0, '/home/chiara/LesionLocator/lesionlocator')
sys.path.insert(0, '/home/chiara/uniGradICON/src')

from dataset_normalized import CTPETDataset, train_val_split, crop_to_ct_shape, CT_DIR, PET_DIR
from preprocessing.resampling.default_resampling import resample_data_or_seg_to_spacing

import unigradicon
from icon_registration.mermaidlite import compute_warped_image_multiNC

PET_LABEL_DIR = "/scratch/nnUNet_raw/Dataset900_USZMelanoma/labelsTr"
LESION_LABEL  = 1          # on garde seulement le label 1 (tumeur)
RESULTS_DIR   = "/home/chiara/results"
UNIGRADICON_SHAPE = (175, 175, 175)

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging():
    log_file = os.path.join(RESULTS_DIR, f"eval_dice_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return log_file


def resize_nearest(tensor):
    """Resize (B,C,D,H,W) vers 175^3 en nearest (préserve un masque binaire)."""
    return F.interpolate(tensor, size=UNIGRADICON_SHAPE, mode='nearest')


def resize_trilinear(tensor):
    """Resize (B,C,D,H,W) vers 175^3 en trilinéaire (images continues CT)."""
    return F.interpolate(tensor, size=UNIGRADICON_SHAPE, mode='trilinear', align_corners=False)


def dice(a, b, eps=1e-8):
    """Dice entre deux masques binaires. 1 = superposition parfaite, 0 = aucune."""
    a = (a.flatten() > 0.5).float()
    b = (b.flatten() > 0.5).float()
    inter = (a * b).sum()
    denom = a.sum() + b.sum()
    if denom < eps:          # deux masques vides -> Dice indéfini, on retourne nan
        return float('nan')
    return float(((2 * inter) / (denom + eps)).item())


def load_lesion_mask(timepoint, patient_id, ct_ref_image):
    """
    Charge le masque de lésion PET, garde le label 1 uniquement, puis applique
    le même resample+crop que le PET (mais en nearest neighbor pour rester binaire).
    Retourne un np.array (D,H,W) float32 dans {0,1}, ou None si le fichier manque.
    """
    path = os.path.join(PET_LABEL_DIR, f"{timepoint}_{patient_id}.nii.gz")
    if not os.path.exists(path):
        return None

    mask_img = sitk.ReadImage(path)
    mask_arr = sitk.GetArrayFromImage(mask_img)

    # Garder SEULEMENT le label 1 (tumeur) -> masque binaire
    mask_bin = (mask_arr == LESION_LABEL).astype(np.float32)

    # Resample au spacing CT (is_seg=True -> nearest, préserve le binaire)
    ct_spacing   = ct_ref_image.GetSpacing()
    mask_spacing = mask_img.GetSpacing()
    mask_4d = mask_bin[np.newaxis, ...]
    mask_resampled = resample_data_or_seg_to_spacing(
        data=mask_4d,
        current_spacing=mask_spacing[::-1],
        new_spacing=ct_spacing[::-1],
        is_seg=True,
    )[0]

    # Crop à la shape CT (comme le PET)
    ct_array = sitk.GetArrayFromImage(ct_ref_image)
    mask_cropped = crop_to_ct_shape(mask_resampled, ct_array)
    return mask_cropped.astype(np.float32)


def evaluate_pair(net, sample):
    patient_id = sample['patient_id']
    tp_source, tp_target = sample['timepoints']

    # CT resizé pour calculer phi
    ct_source = resize_trilinear(sample['ct_source'].unsqueeze(0).to(device))
    ct_target = resize_trilinear(sample['ct_target'].unsqueeze(0).to(device))

    # Charger les masques source & target (espace CT), via l'image CT de référence
    ct_ref = sitk.ReadImage(sample['ct_source_path'])
    mask_src = load_lesion_mask(tp_source, patient_id, ct_ref)
    mask_tgt = load_lesion_mask(tp_target, patient_id, ct_ref)

    if mask_src is None or mask_tgt is None:
        net.clean() if hasattr(net, 'clean') else None
        return None

    # Tensors masques -> 175^3 nearest
    mask_src_t = resize_nearest(torch.from_numpy(mask_src).unsqueeze(0).unsqueeze(0).to(device))
    mask_tgt_t = resize_nearest(torch.from_numpy(mask_tgt).unsqueeze(0).unsqueeze(0).to(device))

    with torch.no_grad():
        net(ct_source, ct_target)          # calcule phi
        phi = net.phi_AB_vectorfield
        spacing = net.spacing

        # Warp du masque source avec phi, en nearest (spline_order=0)
        mask_warped = compute_warped_image_multiNC(
            mask_src_t, phi, spacing, spline_order=0, zero_boundary=True)

    m = {
        'patient_id': patient_id,
        'pair': f"{tp_source}->{tp_target}",
        'n_vox_src': int((mask_src_t > 0.5).sum().item()),
        'n_vox_tgt': int((mask_tgt_t > 0.5).sum().item()),
        'dice_baseline': dice(mask_src_t, mask_tgt_t),      # lésions sans recalage
        'dice_warped':   dice(mask_warped, mask_tgt_t),     # lésions après phi
    }
    if hasattr(net, 'clean'):
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
    logging.info(f"Évaluation Dice sur {len(val_ds)} paires de validation")

    all_metrics = []
    skipped_no_mask = 0
    n = len(val_ds) if args.max_pairs < 0 else min(args.max_pairs, len(val_ds))
    for i in range(n):
        try:
            m = evaluate_pair(net, val_ds[i])
            if m is None:
                skipped_no_mask += 1
                logging.info(f"[{i+1}/{n}] masque manquant, skip")
                continue
            all_metrics.append(m)
            logging.info(
                f"[{i+1}/{n}] {m['patient_id']} {m['pair']} | "
                f"vox src/tgt={m['n_vox_src']}/{m['n_vox_tgt']} | "
                f"Dice {m['dice_baseline']:.3f} -> {m['dice_warped']:.3f}"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logging.warning(f"[{i+1}/{n}] OOM, skip")
                continue
            raise

    def mean_std(key):
        vals = [m[key] for m in all_metrics if m[key] is not None and not np.isnan(m[key])]
        return (np.mean(vals), np.std(vals), len(vals)) if vals else (float('nan'), float('nan'), 0)

    logging.info("=" * 70)
    logging.info("RÉSULTATS DICE LÉSIONS (label 1, moyenne ± écart-type)")
    db_m, db_s, db_n = mean_std('dice_baseline')
    dw_m, dw_s, dw_n = mean_std('dice_warped')
    logging.info(f"Lésions baseline (sans phi): Dice = {db_m:.3f} ± {db_s:.3f}  (n={db_n})")
    logging.info(f"Lésions recalé  (avec phi) : Dice = {dw_m:.3f} ± {dw_s:.3f}  (n={dw_n})")
    logging.info(f"  -> gain Dice             : {dw_m - db_m:+.3f}")
    logging.info(f"  paires sans masque       : {skipped_no_mask}")
    logging.info("=" * 70)
    if dw_m > db_m:
        logging.info("phi calculé sur CT améliore l'alignement des LÉSIONS PET -> hypothèse soutenue")
    else:
        logging.info("phi calculé sur CT n'améliore pas l'alignement des lésions -> hypothèse non soutenue")
    logging.info("=" * 70)

    import csv
    csv_path = os.path.join(RESULTS_DIR, f"dice_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
    with open(csv_path, 'w', newline='') as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
    logging.info(f"CSV : {csv_path}")