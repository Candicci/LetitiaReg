# check_alignment_all_patients.py
import sys, os, csv
sys.path.insert(0, '/home/chiara')

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_normalized import scan_dataset, CT_DIR, PET_DIR

OUT_DIR = '/home/chiara/results'
os.makedirs(OUT_DIR, exist_ok=True)

def read_header(path):
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return r.GetOrigin(), r.GetSize(), r.GetSpacing(), r.GetDirection()

dataset = scan_dataset(CT_DIR, PET_DIR)
rows = []
for pid, tps in dataset.items():
    if 'TP0' not in tps or 'ct' not in tps['TP0'] or 'pet' not in tps['TP0']:
        continue
    ct_o, ct_sz, ct_sp, ct_dir = read_header(tps['TP0']['ct'])
    pet_o, pet_sz, pet_sp, pet_dir = read_header(tps['TP0']['pet'])
    ct_ext  = tuple(s*sp for s, sp in zip(ct_sz, ct_sp))
    pet_ext = tuple(s*sp for s, sp in zip(pet_sz, pet_sp))
    ct_c  = tuple(o + e/2 for o, e in zip(ct_o, ct_ext))
    pet_c = tuple(o + e/2 for o, e in zip(pet_o, pet_ext))
    offset_xy = float(np.hypot(ct_c[0]-pet_c[0], ct_c[1]-pet_c[1]))
    offset_z  = float(abs(ct_c[2]-pet_c[2]))
    rows.append({
        'patient_id': pid,
        'ct_extent_xy_mm': round(ct_ext[0], 1),
        'pet_extent_xy_mm': round(pet_ext[0], 1),
        'fov_diff_mm': round(pet_ext[0]-ct_ext[0], 1),
        'center_offset_xy_mm': round(offset_xy, 2),
        'center_offset_z_mm': round(offset_z, 2),
        'direction_match': ct_dir == pet_dir,
    })

csv_path = os.path.join(OUT_DIR, 'alignment_check_all_patients.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

offsets = np.array([r['center_offset_xy_mm'] for r in rows])
n_dir_mismatch = sum(not r['direction_match'] for r in rows)
print(f"n = {len(rows)} patients vérifiés")
print(f"Décalage centre XY (mm): mean={offsets.mean():.2f}, std={offsets.std():.2f}, max={offsets.max():.2f}")
print(f"Mismatch de direction: {n_dir_mismatch}/{len(rows)}")

order = np.argsort(offsets)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(np.arange(len(offsets)), offsets[order], color="#065A82", width=0.8)
ax.set_ylabel("Décalage centre CT/PET, XY (mm)")
ax.set_xlabel(f"Patients (n={len(offsets)}, triés)")
ax.set_title("Écart de centrage CT/PET — vérifié sur toute la cohorte")
ax.grid(axis='y', alpha=0.3)
for sp_ in ['top', 'right']:
    ax.spines[sp_].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'alignment_check_all_patients.png'), dpi=150)
print("Sauvegardé:", os.path.join(OUT_DIR, 'alignment_check_all_patients.png'))