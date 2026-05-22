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

from preprocessing.resampling.default_resampling import (
    resample_data_or_seg_to_spacing
)

"""
Dataset pour charger les paires CT/PET longitudinales.
- CT: Dataset800_USZMelanoma (channel 0000)
- PET: Dataset900_USZMelanoma (channel 0001)
Gère le resampling PET → spacing CT, split train/val,
et paires bidirectionnelles (T0→T1 et T1→T0).
"""

CT_DIR = "/scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr"
PET_DIR = "/scratch/nnUNet_raw/Dataset900_USZMelanoma/imagesTr"


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse un nom de fichier pour extraire timepoint et patient_id.

    Exemple: 'TP0_008_0000.nii.gz' → ('TP0', '008')
    """
    basename = filename.replace('.nii.gz', '').replace('.nii', '')
    pattern = r'(TP\d+)_(\d+)_\d+'
    match = re.match(pattern, basename)

    if match:
        timepoint = match.group(1)   # 'TP0', 'TP1', 'TP2'
        patient_id = match.group(2)  # '008', '010', etc.
        return timepoint, patient_id
    else:
        raise ValueError(f"Impossible de parser: {filename}")


def scan_dataset(ct_dir: str, pet_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Scanne les dossiers CT et PET, organise par patient et timepoint.

    Returns:
        {
            '008': {
                'TP0': {'ct': '/path/TP0_008_0000.nii.gz', 'pet': '/path/TP0_008_0001.nii.gz'},
                'TP1': {'ct': '/path/TP1_008_0000.nii.gz', 'pet': '/path/TP1_008_0001.nii.gz'},
            },
            '010': {...},
        }
    """
    dataset = defaultdict(lambda: defaultdict(dict))

    # Scanner les CT (Dataset800)
    for filename in os.listdir(ct_dir):
        if not filename.endswith('.nii.gz'):
            continue
        try:
            timepoint, patient_id = parse_filename(filename)
            dataset[patient_id][timepoint]['ct'] = os.path.join(ct_dir, filename)
        except ValueError:
            continue

    # Scanner les PET (Dataset900)
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
    """
    Génère les paires de timepoints consécutifs.
    Si bidirectional=True, on génère aussi les paires inversées
    pour doubler les données d'entraînement.

    Exemple avec bidirectional=True:
        {'TP0': ..., 'TP1': ..., 'TP2': ...}
        → [('TP0','TP1'), ('TP1','TP0'), ('TP1','TP2'), ('TP2','TP1')]

    Exemple avec bidirectional=False:
        → [('TP0','TP1'), ('TP1','TP2')]
    """
    timepoints = sorted(patient_data.keys())
    pairs = []
    for i in range(len(timepoints) - 1):
        pairs.append((timepoints[i], timepoints[i + 1]))      # Forward: T0 → T1
        if bidirectional:
            pairs.append((timepoints[i + 1], timepoints[i]))  # Backward: T1 → T0
    return pairs


def resample_pet_to_ct_spacing(pet_image: sitk.Image, ct_image: sitk.Image) -> np.ndarray:
    """
    Resample le PET vers le spacing du CT.

    Args:
        pet_image: image PET (SimpleITK)
        ct_image: image CT de référence (SimpleITK)

    Returns:
        PET resampled en numpy array (D, H, W)
    """
    ct_spacing = ct_image.GetSpacing()    # (sx, sy, sz)
    pet_spacing = pet_image.GetSpacing()

    # Convertir PET en numpy (D, H, W) → (1, D, H, W)
    pet_array = sitk.GetArrayFromImage(pet_image)
    pet_array_4d = pet_array[np.newaxis, ...]

    # Resample: inverser spacings car numpy est (Z, Y, X) et SimpleITK est (X, Y, Z)
    pet_resampled = resample_data_or_seg_to_spacing(
        data=pet_array_4d,
        current_spacing=pet_spacing[::-1],
        new_spacing=ct_spacing[::-1],
        is_seg=False
    )

    return pet_resampled[0]  # Enlever dimension channel → (D, H, W)


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class CTPETDataset(Dataset):
    """
    Dataset pour les paires CT/PET longitudinales.

    Chaque élément contient pour une paire (source, target):
    - ct_source, ct_target: images CT
    - pet_source, pet_target: images PET (resampled au spacing CT)
    - patient_id, timepoints
    - spacings

    Avec bidirectional=True, chaque paire (T0,T1) génère aussi (T1,T0),
    ce qui double les données d'entraînement.
    """

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

        # Scanner les deux datasets
        print(f"Scanning CT: {ct_dir}")
        print(f"Scanning PET: {pet_dir}")
        self.dataset = scan_dataset(ct_dir, pet_dir)
        print(f"Found {len(self.dataset)} patients total")

        # Filtrer par patient_ids si spécifié (pour train/val split)
        if patient_ids is not None:
            self.dataset = {k: v for k, v in self.dataset.items() if k in patient_ids}
            print(f"Filtered to {len(self.dataset)} patients")

        # Créer la liste des échantillons: (patient_id, (tp_source, tp_target))
        # On ne garde que les paires où CT ET PET existent aux deux timepoints
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

        # Resample PET vers spacing CT
        if self.resample_pet:
            pet_source_array = resample_pet_to_ct_spacing(pet_source, ct_source)
            pet_target_array = resample_pet_to_ct_spacing(pet_target, ct_target)
        else:
            pet_source_array = sitk.GetArrayFromImage(pet_source)
            pet_target_array = sitk.GetArrayFromImage(pet_target)

        ct_source_array = sitk.GetArrayFromImage(ct_source)
        ct_target_array = sitk.GetArrayFromImage(ct_target)

        return {
            'patient_id': patient_id,
            'timepoints': (tp_source, tp_target),
            # Images en tensors (1, D, H, W)
            'ct_source': torch.from_numpy(ct_source_array.astype(np.float32)).unsqueeze(0),
            'ct_target': torch.from_numpy(ct_target_array.astype(np.float32)).unsqueeze(0),
            'pet_source': torch.from_numpy(pet_source_array.astype(np.float32)).unsqueeze(0),
            'pet_target': torch.from_numpy(pet_target_array.astype(np.float32)).unsqueeze(0),
            # Métadonnées
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
    """
    Split les patients en train et validation avec seed fixe.
    Le split est au niveau PATIENT: toutes les paires d'un patient
    sont soit dans train, soit dans val (pas de data leakage).

    Returns:
        (train_patient_ids, val_patient_ids)
    """
    dataset = scan_dataset(ct_dir, pet_dir)

    # Ne garder que les patients avec au moins 2 timepoints
    valid_patients = sorted([
        pid for pid, data in dataset.items()
        if len(data) >= 2
    ])

    # Shuffle avec seed fixe (reproductible)
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_patients)

    # Split
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
    """
    Crée les DataLoaders train et validation.
    batch_size=1 car les images 3D sont volumineuses.
    bidirectional=True pour doubler les paires d'entraînement.
    """
    train_ids, val_ids = train_val_split(val_ratio=val_ratio, seed=seed)

    train_dataset = CTPETDataset(
        patient_ids=train_ids,
        bidirectional=bidirectional
    )
    val_dataset = CTPETDataset(
        patient_ids=val_ids,
        bidirectional=False  # Pas de doublons pour la validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU DATASET")
    print("=" * 60)

    # 1. Scanner
    print("\n1. Scanning datasets...")
    dataset = scan_dataset(CT_DIR, PET_DIR)
    print(f"   {len(dataset)} patients trouvés")

    for i, (pid, data) in enumerate(sorted(dataset.items())[:5]):
        timepoints = sorted(data.keys())
        modalities = {tp: list(data[tp].keys()) for tp in timepoints}
        print(f"   Patient {pid}: {modalities}")

    # 2. Compter les paires (forward only vs bidirectional)
    print("\n2. Paires valides...")
    forward_pairs = 0
    bidir_pairs = 0
    for pid, data in dataset.items():
        pairs_fwd = get_timepoint_pairs(data, bidirectional=False)
        pairs_bi = get_timepoint_pairs(data, bidirectional=True)
        for tp_s, tp_t in pairs_fwd:
            has_ct = 'ct' in data.get(tp_s, {}) and 'ct' in data.get(tp_t, {})
            has_pet = 'pet' in data.get(tp_s, {}) and 'pet' in data.get(tp_t, {})
            if has_ct and has_pet:
                forward_pairs += 1
        for tp_s, tp_t in pairs_bi:
            has_ct = 'ct' in data.get(tp_s, {}) and 'ct' in data.get(tp_t, {})
            has_pet = 'pet' in data.get(tp_s, {}) and 'pet' in data.get(tp_t, {})
            if has_ct and has_pet:
                bidir_pairs += 1
    print(f"   Forward only: {forward_pairs} paires")
    print(f"   Bidirectional: {bidir_pairs} paires (×2)")

    # 3. Test du split
    print("\n3. Test split train/val...")
    train_ids, val_ids = train_val_split(seed=42)

    # 4. Test des DataLoaders
    print("\n4. Test DataLoaders...")
    train_ds = CTPETDataset(patient_ids=train_ids, bidirectional=True)
    val_ds = CTPETDataset(patient_ids=val_ids, bidirectional=False)
    print(f"   Train: {len(train_ds)} paires (bidirectional)")
    print(f"   Val: {len(val_ds)} paires (forward only)")

    # 5. Charger un échantillon
    print("\n5. Test chargement d'un échantillon...")
    try:
        ds = CTPETDataset()
        if len(ds) > 0:
            print("   Chargement en cours (peut prendre 1-2 min)...")
            sample = ds[0]
            print(f"   Patient: {sample['patient_id']}")
            print(f"   Timepoints: {sample['timepoints']}")
            print(f"   CT T0 shape: {sample['ct_source'].shape}")
            print(f"   PET T0 shape: {sample['pet_source'].shape}")
    except Exception as e:
        print(f"   Erreur: {e}")

    print("\n" + "=" * 60)
    print("FIN DU TEST")
    print("=" * 60)



