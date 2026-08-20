import sys
import os
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time 

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/chiara/LesionLocator/lesionlocator')

from preprocessing.resampling.default_resampling import (
    resample_data_or_seg_to_spacing
)
from lesionlocator.preprocessing.normalization.map_channel_name_to_normalization import (
    get_normalization_scheme
)

"""
Dataset pour charger les paires CT/PET longitudinales.
- CT: Dataset800_USZMelanoma (channel 0000)
- PET: Dataset900_USZMelanoma (channel 0001)
Gère le resampling PET → spacing CT, normalisation CT/PET,
split train/val, et paires bidirectionnelles (T0→T1 et T1→T0).
"""

CT_DIR = "/scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr"
PET_DIR = "/scratch/nnUNet_raw/Dataset900_USZMelanoma/imagesTr"

# Stats précalculées sur Dataset800_USZMelanoma (300 CT, voxels > -900 HU)
CT_INTENSITY_PROPERTIES = {
    'mean':            -194.7492,
    'std':              386.7393,
    'percentile_00_5': -898.0000,
    'percentile_99_5': 1125.0000,
}

# Patients exclus : écart de couverture axiale CT/PET > 5mm à TP0
# (cf. check_alignment_all_patients.py — probable différence de protocole d'acquisition,
#  pas un bug de recalage ; exclus par précaution pour éviter du padding silencieux)
Z_MISALIGNED_PATIENTS = {
    '061', '107', '108', '109', '223', '256', '289', '317',
    '384', '394', '396', '397', '401', '445', '452', '470', '476', '492'
}

def parse_filename(filename: str) -> Tuple[str, str]:
    basename = filename.replace('.nii.gz', '').replace('.nii', '')
    pattern = r'(TP\d+)_(\d+)_\d+'
    match = re.match(pattern, basename)
    if match:
        return match.group(1), match.group(2)
    else:
        raise ValueError(f"Impossible de parser: {filename}")


def scan_dataset(ct_dir: str, pet_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    dataset = defaultdict(lambda: defaultdict(dict))
    for filename in os.listdir(ct_dir):
        if not filename.endswith('.nii.gz'):
            continue
        try:
            timepoint, patient_id = parse_filename(filename)
            dataset[patient_id][timepoint]['ct'] = os.path.join(ct_dir, filename)
        except ValueError:
            continue
    for filename in os.listdir(pet_dir):
        if not filename.endswith('.nii.gz'):
            continue
        try:
            timepoint, patient_id = parse_filename(filename)
            dataset[patient_id][timepoint]['pet'] = os.path.join(pet_dir, filename)
        except ValueError:
            continue
    return dict(dataset)


def get_timepoint_pairs(
    patient_data: Dict[str, Dict[str, str]],
    bidirectional: bool = True
) -> List[Tuple[str, str]]:
    timepoints = sorted(patient_data.keys())
    pairs = []
    for i in range(len(timepoints) - 1):
        pairs.append((timepoints[i], timepoints[i + 1]))
        if bidirectional:
            pairs.append((timepoints[i + 1], timepoints[i]))
    return pairs


def resample_pet_to_ct_spacing(pet_image: sitk.Image, ct_image: sitk.Image) -> np.ndarray:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)      # grille CT : même origin/spacing/direction/size
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    pet_resampled = resampler.Execute(pet_image)
    return sitk.GetArrayFromImage(pet_resampled).astype(np.float32)



# =============================================================================
# DATASET PYTORCH
# =============================================================================

class CTPETDataset(Dataset):

    def __init__(
        self,
        ct_dir: str = CT_DIR,
        pet_dir: str = PET_DIR,
        resample_pet: bool = True,
        bidirectional: bool = True,
        patient_ids: Optional[List[str]] = None,
    ):
        self.ct_dir = ct_dir
        self.pet_dir = pet_dir
        self.resample_pet = resample_pet
        self.bidirectional = bidirectional

        print(f"Scanning CT: {ct_dir}")
        print(f"Scanning PET: {pet_dir}")
        self.dataset = scan_dataset(ct_dir, pet_dir)
        print(f"Found {len(self.dataset)} patients total")

        if patient_ids is not None:
            self.dataset = {k: v for k, v in self.dataset.items() if k in patient_ids}
            print(f"Filtered to {len(self.dataset)} patients")

        # Instancier les normaliseurs une seule fois (pas dans __getitem__)
        CTNorm = get_normalization_scheme('ct')
        self.ct_normalizer = CTNorm(
            use_mask_for_norm=False,
            intensityproperties=CT_INTENSITY_PROPERTIES
        )
        PETNorm = get_normalization_scheme('zscore')
        self.pet_normalizer = PETNorm(
            use_mask_for_norm=True,
            intensityproperties={'mean': 0, 'std': 1}
        )

        self.samples = []
        for patient_id, patient_data in self.dataset.items():
            pairs = get_timepoint_pairs(patient_data, bidirectional=self.bidirectional)
            for tp_source, tp_target in pairs:
                has_ct = ('ct' in patient_data.get(tp_source, {}) and
                          'ct' in patient_data.get(tp_target, {}))
                has_pet = ('pet' in patient_data.get(tp_source, {}) and
                           'pet' in patient_data.get(tp_target, {}))
                if has_ct and has_pet:
                    self.samples.append((patient_id, (tp_source, tp_target)))

        direction = "bidirectional" if bidirectional else "forward only"
        print(f"Total valid pairs ({direction}): {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        patient_id, (tp_source, tp_target) = self.samples[idx]
        patient_data = self.dataset[patient_id]

        # Charger les 4 images
        ct_source = sitk.ReadImage(patient_data[tp_source]['ct'])
        ct_target = sitk.ReadImage(patient_data[tp_target]['ct'])
        pet_source = sitk.ReadImage(patient_data[tp_source]['pet'])
        pet_target = sitk.ReadImage(patient_data[tp_target]['pet'])

        # Arrays numpy float32
        ct_source_array = sitk.GetArrayFromImage(ct_source).astype(np.float32)
        ct_target_array = sitk.GetArrayFromImage(ct_target).astype(np.float32)

        # Resample PET → spacing CT
        if self.resample_pet:
            pet_source_array = resample_pet_to_ct_spacing(pet_source, ct_source)
            pet_target_array = resample_pet_to_ct_spacing(pet_target, ct_target)
        else:
            pet_source_array = sitk.GetArrayFromImage(pet_source).astype(np.float32)
            pet_target_array = sitk.GetArrayFromImage(pet_target).astype(np.float32)


        assert ct_source_array.shape == pet_source_array.shape, \
            f"Shape mismatch CT/PET après resampling: {ct_source_array.shape} vs {pet_source_array.shape}"
        assert ct_target_array.shape == pet_target_array.shape, \
            f"Shape mismatch CT/PET après resampling: {ct_target_array.shape} vs {pet_target_array.shape}"
        # =====================================================
        # NORMALISATION CT (stats globales dataset)
        # =====================================================
        ct_source_array = self.ct_normalizer.run(ct_source_array)
        ct_target_array = self.ct_normalizer.run(ct_target_array)

        # =====================================================
        # NORMALISATION PET (z-score sur masque foreground)
        # =====================================================
        pet_source_array = self.pet_normalizer.run(pet_source_array)
        pet_target_array = self.pet_normalizer.run(pet_target_array)

        # Assertions de cohérence
        assert ct_source_array.dtype == np.float32
        assert pet_source_array.dtype == np.float32
        assert not np.isnan(ct_source_array).any(), "NaN dans CT source!"
        assert not np.isnan(pet_source_array).any(), "NaN dans PET source!"

        return {
            'patient_id': patient_id,
            'timepoints': (tp_source, tp_target),
            'ct_source':  torch.from_numpy(ct_source_array).unsqueeze(0),
            'ct_target':  torch.from_numpy(ct_target_array).unsqueeze(0),
            'pet_source': torch.from_numpy(pet_source_array).unsqueeze(0),
            'pet_target': torch.from_numpy(pet_target_array).unsqueeze(0),
            'ct_source_spacing': ct_source.GetSpacing(),
            'ct_target_spacing': ct_target.GetSpacing(),
            'ct_source_path': patient_data[tp_source]['ct'],
            'ct_target_path': patient_data[tp_target]['ct'],
        }


# =============================================================================
# TRAIN/VAL SPLIT
# =============================================================================

def train_val_split(
    ct_dir: str = CT_DIR,
    pet_dir: str = PET_DIR,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    dataset = scan_dataset(ct_dir, pet_dir)
    valid_patients = sorted([
        pid for pid, data in dataset.items()
        if len(data) >= 2 and pid not in Z_MISALIGNED_PATIENTS
    ])
    print(f"Exclus pour couverture Z incohérente: {len(Z_MISALIGNED_PATIENTS)} patients")
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_patients)
    n_val = int(len(valid_patients) * val_ratio)
    val_ids = valid_patients[:n_val]
    train_ids = valid_patients[n_val:]
    print(f"Split (seed={seed}): {len(train_ids)} train, {len(val_ids)} val patients")
    return train_ids, val_ids


def get_dataloaders(
    batch_size: int = 1,
    val_ratio: float = 0.2,
    seed: int = 42,
    bidirectional: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    train_ids, val_ids = train_val_split(val_ratio=val_ratio, seed=seed)
    train_dataset = CTPETDataset(patient_ids=train_ids, bidirectional=bidirectional)
    val_dataset = CTPETDataset(patient_ids=val_ids, bidirectional=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU DATASET")
    print("=" * 60)

    print("\n1. Scanning datasets...")
    dataset = scan_dataset(CT_DIR, PET_DIR)
    print(f"   {len(dataset)} patients trouvés")

    print("\n2. Test split train/val...")
    train_ids, val_ids = train_val_split(seed=42)

    print("\n3. Test DataLoaders...")
    train_ds = CTPETDataset(patient_ids=train_ids, bidirectional=True)
    val_ds = CTPETDataset(patient_ids=val_ids, bidirectional=False)
    print(f"   Train: {len(train_ds)} paires")
    print(f"   Val:   {len(val_ds)} paires")

    print("\n4. Test chargement + normalisation sur patient 008...")
    try:
        ds = CTPETDataset(patient_ids=['008'], bidirectional=False, resample_pet=True)
        sample = ds[0]
        print(f"   Patient:    {sample['patient_id']}")
        print(f"   Timepoints: {sample['timepoints']}")
        print(f"   CT source  shape: {sample['ct_source'].shape}, "
              f"min={sample['ct_source'].min():.2f}, max={sample['ct_source'].max():.2f}")
        print(f"   PET source shape: {sample['pet_source'].shape}, "
              f"min={sample['pet_source'].min():.2f}, max={sample['pet_source'].max():.2f}")
        print("   ✅ Normalisation OK !")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    print("\n" + "=" * 60)
    print("FIN DU TEST")
    print("=" * 60)
    
    
    
