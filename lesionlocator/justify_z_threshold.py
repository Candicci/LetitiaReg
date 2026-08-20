# justify_z_threshold.py
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv('results/alignment_check_all_patients.csv')
vals = np.sort(df['center_offset_z_mm'].values)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(np.arange(len(vals)), vals, marker='o', markersize=3, color="#065A82")
ax.axhline(5, color='#B9C4CC', linestyle='--', label='seuil choisi (5mm)')
ax.set_yscale('symlog')  # utile vu l'écart d'échelle 0.01mm à 80mm
ax.set_xlabel('Patients (triés par offset Z croissant)')
ax.set_ylabel('Offset centre Z (mm, échelle log)')
ax.set_title("Rupture naturelle dans la distribution des offsets Z\n(justifie le seuil d'exclusion, pas un choix arbitraire)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('z_threshold_justification.png', dpi=150)