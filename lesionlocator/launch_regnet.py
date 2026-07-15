"""
inspect_regnet_ckpt.py
======================
Inspecte le checkpoint RegNet fourni par la prof pour comprendre :
  - sa structure (state_dict brut ? dict avec model_state_dict ?)
  - les noms de couches → est-ce la même architecture que uniGradICON ?
  - de quoi savoir COMMENT le charger

Aucun calcul lourd, juste torch.load sur CPU. À lancer avant d'écrire le
code de chargement définitif.
"""

import torch
import sys

CKPT = "/scratch/LesionLocator_saved_ckpt/TrainSeg900_LesionLocatorFTDec/LesionLocatorTrack/fold_1/best_tracking_model.pth"

print("=" * 70)
print("INSPECTION DU CHECKPOINT REGNET")
print("=" * 70)
print(f"Chemin : {CKPT}\n")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)

print(f"Type de l'objet chargé : {type(ckpt)}\n")

# Cas 1 : c'est un dict de haut niveau (avec epoch, config, etc.)
if isinstance(ckpt, dict):
    print("Clés de premier niveau :")
    for k in ckpt.keys():
        v = ckpt[k]
        vtype = type(v).__name__
        extra = ""
        if isinstance(v, (int, float, str)):
            extra = f" = {v}"
        elif isinstance(v, dict):
            extra = f" ({len(v)} entrées)"
        print(f"   {k:30s} [{vtype}]{extra}")
    print()

    # Trouver le state_dict des poids
    sd = None
    for candidate in ['network_weights', 'model_state_dict', 'state_dict',
                      'network', 'model', 'net', 'regis_net']:
        if candidate in ckpt and isinstance(ckpt[candidate], dict):
            sd = ckpt[candidate]
            print(f"→ state_dict trouvé sous la clé '{candidate}'\n")
            break
    if sd is None:
        # Peut-être que ckpt EST déjà le state_dict
        if all(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:5]):
            sd = ckpt
            print("→ le dict lui-même semble être le state_dict\n")
else:
    sd = ckpt
    print("→ l'objet chargé est directement le state_dict\n")

# Analyser les noms de couches
if sd is not None and isinstance(sd, dict):
    keys = list(sd.keys())
    print(f"Nombre de tensors dans le state_dict : {len(keys)}\n")
    print("30 premiers noms de couches :")
    for k in keys[:30]:
        shape = tuple(sd[k].shape) if isinstance(sd[k], torch.Tensor) else "?"
        print(f"   {k:55s} {shape}")
    print()

    # Indices d'architecture uniGradICON
    joined = " ".join(keys).lower()
    print("Indices d'architecture :")
    for marker in ['regis_net', 'phi', 'unet', 'downsample', 'net_a', 'netphi',
                   'twostep', 'gradicon', 'displacement']:
        present = "OUI" if marker in joined else "non"
        print(f"   contient '{marker}' : {present}")
    print()

    # Préfixe commun (utile pour le remapping des clés)
    prefixes = set(k.split('.')[0] for k in keys)
    print(f"Préfixes de premier niveau : {sorted(prefixes)}")

print("=" * 70)
print("Copie-colle cette sortie pour qu'on écrive le bon code de chargement.")
print("=" * 70)