# recompute_paired_comparison.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BASE, C_CT, C_PET = "#B9C4CC", "#065A82", "#1C7293"

df = pd.read_csv('twopass_20260820-181359.csv')

print(f"Total paires évaluées (CT-transfer vs baseline) : {len(df)}")

# --- Comparaison faible : baseline vs CT-transfer, sur TOUTES les paires disponibles ---
print(f"\nBaseline      : {df['ncc_baseline'].mean():.3f} ± {df['ncc_baseline'].std():.3f}  (n={df['ncc_baseline'].notna().sum()})")
print(f"CT-transfer   : {df['ncc_ct_transfer'].mean():.3f} ± {df['ncc_ct_transfer'].std():.3f}  (n={df['ncc_ct_transfer'].notna().sum()})")
print(f"Gain moyen    : {df['ncc_ct_transfer'].mean() - df['ncc_baseline'].mean():+.3f}")

# --- Comparaison forte : CT-transfer vs PET-native, UNIQUEMENT sur les paires où les deux existent ---
paired = df.dropna(subset=['ncc_pet_native']).copy()
print(f"\n--- Comparaison appariée (n={len(paired)} paires avec PET-native disponible) ---")
print(f"CT-transfer (sous-ensemble apparié) : {paired['ncc_ct_transfer'].mean():.3f} ± {paired['ncc_ct_transfer'].std():.3f}")
print(f"PET-native                          : {paired['ncc_pet_native'].mean():.3f} ± {paired['ncc_pet_native'].std():.3f}")
print(f"Diff (CT-transfer - PET-native)     : {paired['ncc_ct_transfer'].mean() - paired['ncc_pet_native'].mean():+.3f}")

# --- Figure scatter corrigée, sur le bon sous-ensemble ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(paired['ncc_pet_native'], paired['ncc_ct_transfer'], color=C_CT, alpha=0.8, edgecolor='white', s=90)
for _, row in paired.iterrows():
    ax.annotate(row['patient_id'], (row['ncc_pet_native'], row['ncc_ct_transfer']),
                fontsize=8, xytext=(4, 4), textcoords='offset points')
ax.plot([0, 1], [0, 1], '--', color='#888', label='y = x (égalité)')
ax.set_xlabel('PET-native NCC (RegNet)')
ax.set_ylabel('CT-transfer NCC (notre méthode)')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
ax.legend(loc='lower right')
ax.set_title(f'CT-transfer vs PET-native — comparaison appariée (n={len(paired)})')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('scatter_paired_corrected.png', dpi=150)
print("\nFigure sauvegardée : scatter_paired_corrected.png")