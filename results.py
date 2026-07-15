"""
Figure des résultats NCC — hypothèse LETITIA (modèle pré-entraîné).
Données : 29 paires de validation, obtenues via evaluate_ncc.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Données brutes (patient, pair, ncc_ct_control, ncc_pet_baseline, ncc_pet_warped)
data = [
    ("008", "TP0->TP1", 0.691, 0.719, 0.669),
    ("008", "TP1->TP2", 0.719, 0.518, 0.720),
    ("029", "TP0->TP1", 0.766, 0.629, 0.692),
    ("029", "TP1->TP2", 0.720, 0.088, 0.710),
    ("040", "TP0->TP1", 0.670, 0.572, 0.499),
    ("040", "TP1->TP2", 0.725, 0.543, 0.533),
    ("055", "TP0->TP1", 0.817, 0.405, 0.618),
    ("064", "TP0->TP1", 0.770, 0.626, 0.722),
    ("064", "TP1->TP2", 0.837, 0.518, 0.543),
    ("091", "TP0->TP1", 0.744, 0.089, 0.587),
    ("091", "TP1->TP2", 0.751, 0.095, 0.471),
    ("098", "TP0->TP1", 0.637, 0.330, 0.374),
    ("098", "TP1->TP2", 0.802, 0.329, 0.636),
    ("113", "TP0->TP1", 0.760, 0.821, 0.753),
    ("113", "TP1->TP2", 0.718, 0.612, 0.672),
    ("116", "TP0->TP1", 0.696, 0.291, 0.550),
    ("116", "TP1->TP2", 0.837, 0.179, 0.484),
    ("133", "TP0->TP1", 0.835, 0.504, 0.728),
    ("133", "TP1->TP2", 0.669, 0.389, 0.632),
    ("150", "TP0->TP1", 0.721, 0.601, 0.660),
    ("150", "TP1->TP2", 0.783, 0.227, 0.315),
    ("246", "TP0->TP1", 0.771, 0.178, 0.612),
    ("246", "TP1->TP2", 0.776, 0.390, 0.700),
    ("256", "TP0->TP1", 0.699, 0.144, 0.190),
    ("352", "TP0->TP1", 0.722, 0.512, 0.403),
    ("406", "TP1->TP2", 0.752, 0.221, 0.571),
    ("452", "TP0->TP1", 0.638, 0.142, 0.532),
    ("452", "TP1->TP2", 0.687, 0.214, 0.487),
    ("465", "TP0->TP2", 0.602, 0.273, 0.290),
]

labels    = [f"{p}\n{pair.replace('->','→')}" for p, pair, *_ in data]
baseline  = np.array([d[3] for d in data])
warped    = np.array([d[4] for d in data])
ct_ctrl   = np.array([d[2] for d in data])
gain      = warped - baseline

# Trier par baseline croissant pour lisibilité
order = np.argsort(baseline)
labels_s   = [labels[i] for i in order]
baseline_s = baseline[order]
warped_s   = warped[order]
gain_s     = gain[order]

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2.2, 1]}
)

# ---- Panneau 1 : baseline vs warped, par paire ----
x = np.arange(len(data))
w = 0.4
ax1.bar(x - w/2, baseline_s, w, label='PET baseline (no φ)', color='#c0c0c0')
ax1.bar(x + w/2, warped_s,  w, label='PET after φ (from CT)', color='#2b8cbe')

ax1.axhline(baseline.mean(), color='#888', ls='--', lw=1,
            label=f'mean baseline = {baseline.mean():.3f}')
ax1.axhline(warped.mean(), color='#08589e', ls='--', lw=1,
            label=f'mean after φ = {warped.mean():.3f}')

ax1.set_ylabel('NCC (higher = better alignment)', fontsize=11)
ax1.set_title('PET alignment before vs after applying CT-derived deformation field φ\n'
              '29 validation pairs · pretrained uniGradICON · sorted by baseline',
              fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels_s, fontsize=6.5, rotation=90)
ax1.legend(loc='upper left', fontsize=9)
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3)

# ---- Panneau 2 : gain par paire ----
colors = ['#2ca25f' if g > 0 else '#de2d26' for g in gain_s]
ax2.bar(x, gain_s, color=colors)
ax2.axhline(0, color='black', lw=0.8)
ax2.axhline(gain.mean(), color='#08589e', ls='--', lw=1.2,
            label=f'mean gain = +{gain.mean():.3f}')
ax2.set_ylabel('NCC gain (φ − baseline)', fontsize=11)
ax2.set_xlabel('validation pair', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(labels_s, fontsize=6.5, rotation=90)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)
legend_elems = [Patch(facecolor='#2ca25f', label='improved'),
                Patch(facecolor='#de2d26', label='degraded')]
ax2.legend(handles=legend_elems + [plt.Line2D([0],[0], color='#08589e', ls='--',
           label=f'mean gain = +{gain.mean():.3f}')], loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('/home/chiara/LesionLocator/lesionlocator/ncc_results.png', dpi=150, bbox_inches='tight')
print("Figure saved")

# Résumé chiffré
n_improved = int((gain > 0).sum())
print(f"CT control NCC  : {ct_ctrl.mean():.3f} ± {ct_ctrl.std():.3f}")
print(f"PET baseline    : {baseline.mean():.3f} ± {baseline.std():.3f}")
print(f"PET after phi   : {warped.mean():.3f} ± {warped.std():.3f}")
print(f"Mean gain       : +{gain.mean():.3f}")
print(f"Improved pairs  : {n_improved}/{len(data)}")
print(f"Std reduction   : {baseline.std():.3f} -> {warped.std():.3f}")