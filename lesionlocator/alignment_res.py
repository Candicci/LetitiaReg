# generate_alignment_proof.py
import sys, os
sys.path.insert(0, '/home/chiara')
sys.path.insert(0, '/home/chiara/LesionLocator/lesionlocator')

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_normalized import resample_pet_to_ct_spacing  # version CORRIGÉE
from preprocessing.resampling.default_resampling import resample_data_or_seg_to_spacing

PID = '008'
CT_PATH  = f'/scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr/TP0_{PID}_0000.nii.gz'
PET_PATH = f'/scratch/nnUNet_raw/Dataset900_USZMelanoma/imagesTr/TP0_{PID}_0001.nii.gz'
OUT_DIR = '/home/chiara/results'
os.makedirs(OUT_DIR, exist_ok=True)

ct  = sitk.ReadImage(CT_PATH)
pet = sitk.ReadImage(PET_PATH)
ct_array = sitk.GetArrayFromImage(ct)

# --- ANCIENNE méthode (spacing seul, bug) ---
pet_spacing_only = resample_data_or_seg_to_spacing(
    data=sitk.GetArrayFromImage(pet)[np.newaxis, ...],
    current_spacing=pet.GetSpacing()[::-1],
    new_spacing=ct.GetSpacing()[::-1],
    is_seg=False
)[0]

# --- NOUVELLE méthode (grille CT complète) ---
pet_fixed = resample_pet_to_ct_spacing(pet, ct)

print("CT shape:            ", ct_array.shape)
print("PET ancien (spacing seul):", pet_spacing_only.shape, "<- mismatch avec CT, nécessitait un crop manuel")
print("PET corrigé (grille CT):  ", pet_fixed.shape, "<- match exact avec CT")

mid = ct_array.shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
axes[0].imshow(ct_array[mid], cmap='gray')
axes[0].set_title(f'CT — patient {PID}\nshape {ct_array.shape}')

axes[1].imshow(ct_array[mid], cmap='gray')
h, w = ct_array.shape[1], ct_array.shape[2]
ph, pw = pet_spacing_only.shape[1], pet_spacing_only.shape[2]
sh, sw = max((ph-h)//2, 0), max((pw-w)//2, 0)
pet_cropped_hack = pet_spacing_only[:, sh:sh+h, sw:sw+w]
axes[1].imshow(pet_cropped_hack[mid], cmap='hot', alpha=0.5)
axes[1].set_title('AVANT — resample spacing seul\n+ crop manuel au centre (hack)')

axes[2].imshow(ct_array[mid], cmap='gray')
axes[2].imshow(pet_fixed[mid], cmap='hot', alpha=0.5)
axes[2].set_title('APRÈS — sitk.Resample(ReferenceImage=CT)\nalignement physique, sans hack')

for ax in axes:
    ax.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'alignment_proof_before_after.png'), dpi=150)
print("Sauvegardé:", os.path.join(OUT_DIR, 'alignment_proof_before_after.png'))