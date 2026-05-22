import SimpleITK as sitk
import numpy as np

# Import du resampling de lesionlocator
from lesionlocator.preprocessing.resampling.default_resampling import (
   resample_data_or_seg_to_spacing
)

def fmt_spacing(s):
    return f"({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f})"


# === CHARGER CT et PET ===
img_ct = sitk.ReadImage('/scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr/TP0_008_0000.nii.gz')
img_pet = sitk.ReadImage('/scratch/nnUNet_raw/Dataset900_USZMelanoma/imagesTr/TP0_008_0001.nii.gz')

# === VOIR LES SPACINGS ===
print(f"CT  spacing: {fmt_spacing(img_ct.GetSpacing())}")
print(f"PET spacing: {fmt_spacing(img_pet.GetSpacing())}")

# === RESAMPLE PET → SPACING CT ===
ct_spacing = img_ct.GetSpacing()
pet_spacing = img_pet.GetSpacing()

# Convertir PET en numpy (D, H, W) → (1, D, H, W)
pet_array = sitk.GetArrayFromImage(img_pet)
pet_array_4d = pet_array[np.newaxis, : , : , :]  # Ajoute channel

# Resample
pet_resampled = resample_data_or_seg_to_spacing(
   data=pet_array_4d,
   current_spacing=pet_spacing[::-1],  # Inverser car numpy est (D,H,W)
   new_spacing=ct_spacing[::-1],
   is_seg=False
)

print("\n=== AFTER resample ===")
print("PET original shape:", pet_array.shape)
print("PET resampled shape:", pet_resampled[0].shape)
print("CT shape:", sitk.GetArrayFromImage(img_ct).shape)

import matplotlib.pyplot as plt

ct_array = sitk.GetArrayFromImage(img_ct)
mid_slice = ct_array.shape[0] // 2

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 1. CT seul
axes[0].imshow(ct_array[mid_slice], cmap='gray')
axes[0].set_title(f'CT\n{ct_array.shape}')

# 2. PET original
mid_pet = pet_array.shape[0] // 2
axes[1].imshow(pet_array[mid_pet], cmap='hot')
axes[1].set_title(f'PET original\n{pet_array.shape}')

# 3. PET resampled
axes[2].imshow(pet_resampled[0, mid_slice], cmap='hot')
axes[2].set_title(f'PET resampled\n{pet_resampled[0].shape}')

# 4. Overlay (cropper au centre pour matcher les tailles)
ct_h, ct_w = ct_array.shape[1], ct_array.shape[2]
pet_h, pet_w = pet_resampled.shape[2], pet_resampled.shape[3]

# Crop le PET resampled au centre pour matcher la taille du CT
start_h = (pet_h - ct_h) // 2
start_w = (pet_w - ct_w) // 2
pet_cropped = pet_resampled[0, :, start_h:start_h+ct_h, start_w:start_w+ct_w]

axes[3].imshow(ct_array[mid_slice], cmap='gray')
axes[3].imshow(pet_cropped[mid_slice], cmap='hot', alpha=0.5)
axes[3].set_title('Overlay CT + PET')

plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
plt.close()
print("✅ Sauvegardé: comparison.png")

#==================================

# === CHARGER CT_T1 (2ème timepoint) ===
img_ct_t1 = sitk.ReadImage('/scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr/TP1_008_0000.nii.gz')
ct_t1_array = sitk.GetArrayFromImage(img_ct_t1)

print(f"\nCT_T0 shape: {ct_array.shape}, spacing: {fmt_spacing(img_ct.GetSpacing())}")
print(f"CT_T1 shape: {ct_t1_array.shape}, spacing: {fmt_spacing(img_ct_t1.GetSpacing())}")

# === PLOT CT_T0 vs CT_T1 ===
mid = ct_array.shape[0] // 2
mid_t1 = ct_t1_array.shape[0] // 2

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(ct_array[mid], cmap='gray')
axes[0].set_title(f'CT_T0 - Slice {mid}')
axes[1].imshow(ct_t1_array[mid_t1], cmap='gray')
axes[1].set_title(f'CT_T1 - Slice {mid_t1}')
plt.savefig('ct_t0_vs_t1.png', dpi=150)
plt.close()
print("✅ Sauvegardé: ct_t0_vs_t1.png")