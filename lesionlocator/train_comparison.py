"""
evaluate_two_pass.py  —  comparaison 3-voies SANS OOM (two-pass)
================================================================
Le V100 (~15 Go) ne tient pas les DEUX modèles à la fois (uniGradICON + RegNet).
Solution : deux passes séquentielles, un seul modèle sur le GPU à la fois.

  PASSE 1 : uniGradICON  → baseline + CT-transfer (+ contrôle CT) sur toutes les paires
  [libération complète du GPU]
  PASSE 2 : RegNet       → PET-native sur toutes les paires
  FUSION  : combine par paire → figures + CSV

Résultat identique à evaluate_and_plot.py, mais sans dépasser la mémoire.
"""

import sys, os, argparse, logging, csv, gc
from datetime import datetime
import numpy as np
import torch

# cuDNN désactivé : conv3d ne trouve pas d'engine sur ce V100 sinon
torch.backends.cudnn.enabled = False

import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/chiara/LesionLocator/lesionlocator')
sys.path.insert(0, '/home/chiara/uniGradICON/src')

from dataset_normalized import CTPETDataset, train_val_split, CT_DIR, PET_DIR
import unigradicon
from icon_registration.mermaidlite import compute_warped_image_multiNC

REGNET_CKPT = "/scratch/LesionLocator_saved_ckpt/TrainSeg900_LesionLocatorFTDec/LesionLocatorTrack/fold_1/best_tracking_model.pth"
RESULTS_DIR = "/home/chiara/results"
UNIGRADICON_SHAPE = (175, 175, 175)
os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_BASE, C_CT, C_PET = "#B9C4CC", "#065A82", "#1C7293"


def setup_logging():
    lf = os.path.join(RESULTS_DIR, f"twopass_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(lf), logging.StreamHandler(sys.stdout)])
    return lf


def resize(t):
    return F.interpolate(t, size=UNIGRADICON_SHAPE, mode='trilinear', align_corners=False)


def ncc(a, b, eps=1e-8):
    a = a.flatten().float(); b = b.flatten().float()
    a = (a - a.mean()) / (a.std() + eps)
    b = (b - b.mean()) / (b.std() + eps)
    return float((a * b).mean().item())


def free_gpu(*objs):
    """Libère explicitement des objets + vide le cache GPU."""
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def load_regnet(path):
    """RegNet = uniGradICON préfixé 'reg_net.regis_net.' (+ U-Net de seg ignoré)."""
    logging.info(f"Chargement RegNet: {path}")
    ck = torch.load(path, map_location='cpu', weights_only=False)
    full_sd = ck['network_weights']
    reg_sd = {k[len('reg_net.'):]: v for k, v in full_sd.items()
              if k.startswith('reg_net.regis_net.')}
    logging.info(f"  {len(reg_sd)} tensors de recalage extraits")
    net = unigradicon.get_unigradicon()
    missing, unexpected = net.load_state_dict(reg_sd, strict=False)
    frac = (len(reg_sd) - len(unexpected)) / max(len(net.state_dict()), 1)
    logging.info(f"  chargé : {frac:.0%}")
    if frac < 0.5:
        raise RuntimeError("Chargement RegNet incomplet")
    net = net.to(device); net.eval()
    return net


# =============================================================================
# PASSE 1 — uniGradICON : baseline + CT-transfer
# =============================================================================
def pass1_ct(val_ds, n):
    logging.info("=" * 60)
    logging.info("PASSE 1/2 : uniGradICON (baseline + CT-transfer)")
    logging.info("=" * 60)
    net = unigradicon.get_unigradicon().to(device); net.eval()

    results = {}
    for i in range(n):
        torch.cuda.empty_cache()          # repartir propre à chaque paire
        try:
            s = val_ds[i]
            pid, (tp_s, tp_t) = s['patient_id'], s['timepoints']
            key = (pid, tp_s, tp_t)

            ct_s  = resize(s['ct_source'].unsqueeze(0).to(device))
            ct_t  = resize(s['ct_target'].unsqueeze(0).to(device))
            pet_s = resize(s['pet_source'].unsqueeze(0).to(device))
            pet_t = resize(s['pet_target'].unsqueeze(0).to(device))

            with torch.no_grad():
                base = ncc(pet_s, pet_t)
                net(ct_s, ct_t)
                phi, sp = net.phi_AB_vectorfield, net.spacing
                warp_pet = compute_warped_image_multiNC(pet_s, phi, sp, spline_order=1, zero_boundary=True)
                ct_tr = ncc(warp_pet, pet_t); del warp_pet
                warp_ct = compute_warped_image_multiNC(ct_s, phi, sp, spline_order=1, zero_boundary=True)
                ct_ctrl = ncc(warp_ct, ct_t); del warp_ct
                del phi
                if hasattr(net, 'clean'): net.clean()

            results[key] = {'patient_id': pid, 'pair': f"{tp_s}->{tp_t}",
                            'ncc_baseline': base, 'ncc_ct_transfer': ct_tr, 'ncc_ct_control': ct_ctrl}
            logging.info(f"  [{i+1}/{n}] {pid} {tp_s}->{tp_t} | ctrl={ct_ctrl:.3f} base={base:.3f} CT->PET={ct_tr:.3f}")
            free_gpu(ct_s, ct_t, pet_s, pet_t)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); logging.warning(f"  [{i+1}/{n}] OOM skip"); continue
            raise

    free_gpu(net)
    logging.info("Passe 1 terminée, GPU libéré.")
    return results


# =============================================================================
# PASSE 2 — RegNet : PET-native
# =============================================================================
def pass2_pet(val_ds, n, results):
    logging.info("=" * 60)
    logging.info("PASSE 2/2 : RegNet (PET-native)")
    logging.info("=" * 60)
    try:
        net = load_regnet(REGNET_CKPT)
    except Exception as e:
        logging.warning(f"RegNet non chargé ({e}). PET-native indisponible.")
        for k in results: results[k]['ncc_pet_native'] = None
        return results

    for i in range(n):
        torch.cuda.empty_cache()          # repartir propre à chaque paire
        try:
            s = val_ds[i]
            pid, (tp_s, tp_t) = s['patient_id'], s['timepoints']
            key = (pid, tp_s, tp_t)
            if key not in results:      # paire skippée en passe 1
                continue

            pet_s = resize(s['pet_source'].unsqueeze(0).to(device))
            pet_t = resize(s['pet_target'].unsqueeze(0).to(device))
            with torch.no_grad():
                net(pet_s, pet_t)
                phi, sp = net.phi_AB_vectorfield, net.spacing
                warp_pet = compute_warped_image_multiNC(pet_s, phi, sp, spline_order=1, zero_boundary=True)
                pet_nat = ncc(warp_pet, pet_t); del warp_pet, phi
                if hasattr(net, 'clean'): net.clean()
            results[key]['ncc_pet_native'] = pet_nat
            logging.info(f"  [{i+1}/{n}] {pid} {tp_s}->{tp_t} | PET-native={pet_nat:.3f}")
            free_gpu(pet_s, pet_t)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); logging.warning(f"  [{i+1}/{n}] OOM skip")
                results[key]['ncc_pet_native'] = None; continue
            raise

    free_gpu(net)
    return results


# =============================================================================
# FIGURES
# =============================================================================
def make_figures(metrics, out_prefix):
    base = np.array([m['ncc_baseline'] for m in metrics])
    ct   = np.array([m['ncc_ct_transfer'] for m in metrics])
    have = all(m.get('ncc_pet_native') is not None for m in metrics)
    pet  = np.array([m['ncc_pet_native'] for m in metrics]) if have else None
    order = np.argsort(base)

    # barres
    fig, ax = plt.subplots(figsize=(13, 5)); x = np.arange(len(metrics))
    if have:
        w = 0.27
        ax.bar(x-w, base[order], w, label='baseline (no reg)', color=C_BASE)
        ax.bar(x,   ct[order],   w, label='CT-transfer (ours)', color=C_CT)
        ax.bar(x+w, pet[order],  w, label='PET-native (RegNet)', color=C_PET)
    else:
        w = 0.4
        ax.bar(x-w/2, base[order], w, label='baseline', color=C_BASE)
        ax.bar(x+w/2, ct[order],   w, label='CT-transfer', color=C_CT)
    ax.set_ylabel('NCC (higher = better)'); ax.set_ylim(0,1); ax.set_xticks([])
    ax.set_xlabel('validation pairs (sorted by baseline)'); ax.legend(loc='upper left')
    ax.set_title('PET alignment: baseline vs CT-transfer' + (' vs PET-native' if have else ''))
    ax.grid(axis='y', alpha=0.3)
    for sp_ in ['top','right']: ax.spines[sp_].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_bars.png", dpi=150); plt.close(fig)

    # boxplot
    fig, ax = plt.subplots(figsize=(7,5))
    data = [base, ct] + ([pet] if have else [])
    labels = ['baseline','CT-transfer'] + (['PET-native'] if have else [])
    cols = [C_BASE, C_CT] + ([C_PET] if have else [])
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch,c in zip(bp['boxes'], cols): patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_ylabel('NCC'); ax.set_ylim(0,1); ax.set_title('Distribution of PET NCC')
    ax.grid(axis='y', alpha=0.3)
    for sp_ in ['top','right']: ax.spines[sp_].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_box.png", dpi=150); plt.close(fig)

    # scatter CT-transfer vs PET-native
    if have:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(pet, ct, color=C_CT, alpha=0.8, edgecolor='white', s=70)
        ax.plot([0,1],[0,1],'--',color='#888',label='y = x (equal)')
        ax.set_xlabel('PET-native NCC (RegNet)'); ax.set_ylabel('CT-transfer NCC (ours)')
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.legend(loc='lower right')
        ax.set_title('CT-transfer vs PET-native\n(on/above line = as good or better)')
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{out_prefix}_scatter.png", dpi=150); plt.close(fig)
    logging.info(f"Figures : {out_prefix}_bars.png, _box.png" + (", _scatter.png" if have else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=-1)
    args = ap.parse_args()

    setup_logging()
    logging.info(f"Device: {device} | cuDNN enabled: {torch.backends.cudnn.enabled}")

    _, val_ids = train_val_split(ct_dir=CT_DIR, pet_dir=PET_DIR, seed=42)
    val_ds = CTPETDataset(patient_ids=val_ids, bidirectional=False, resample_pet=True)
    n = len(val_ds) if args.max_pairs < 0 else min(args.max_pairs, len(val_ds))
    logging.info(f"Évaluation sur {n} paires (two-pass)")

    results = pass1_ct(val_ds, n)
    results = pass2_pet(val_ds, n, results)

    metrics = list(results.values())
    if not metrics:
        logging.error("Aucune paire évaluée."); sys.exit(1)

    def ms(k):
        v = [m[k] for m in metrics if m.get(k) is not None]
        return (np.mean(v), np.std(v)) if v else (float('nan'), float('nan'))

    b, ct_, pn, cc = ms('ncc_baseline'), ms('ncc_ct_transfer'), ms('ncc_pet_native'), ms('ncc_ct_control')
    logging.info("=" * 60)
    logging.info("RÉSULTATS FINAUX (NCC, moyenne ± std)")
    logging.info(f"  CT control              : {cc[0]:.3f} ± {cc[1]:.3f}")
    logging.info(f"  (1) baseline            : {b[0]:.3f} ± {b[1]:.3f}")
    logging.info(f"  (2) CT-transfer (ours)  : {ct_[0]:.3f} ± {ct_[1]:.3f}")
    if not np.isnan(pn[0]):
        logging.info(f"  (3) PET-native (RegNet) : {pn[0]:.3f} ± {pn[1]:.3f}")
        logging.info(f"  gain (2)-(1)            : {ct_[0]-b[0]:+.3f}")
        logging.info(f"  diff (2)-(3)            : {ct_[0]-pn[0]:+.3f}")
        if ct_[0] >= pn[0] - 0.02:
            logging.info("  → CT-transfer ÉGALE/DÉPASSE PET-native : hypothèse soutenue")
        else:
            logging.info("  → PET-native meilleur : CT-transfer utile mais inférieur")
    logging.info("=" * 60)

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    csvp = os.path.join(RESULTS_DIR, f"twopass_{stamp}.csv")
    with open(csvp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=metrics[0].keys()); w.writeheader(); w.writerows(metrics)
    logging.info(f"CSV : {csvp}")
    make_figures(metrics, os.path.join(RESULTS_DIR, f"twopass_{stamp}"))
    logging.info("Terminé.")