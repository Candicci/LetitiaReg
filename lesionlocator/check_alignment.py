import SimpleITK as sitk

for pid in ['008', '010']:
    ct = sitk.ReadImage(f'/scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr/TP0_{pid}_0000.nii.gz')
    pet = sitk.ReadImage(f'/scratch/nnUNet_raw/Dataset900_USZMelanoma/imagesTr/TP0_{pid}_0001.nii.gz')
    print(f"--- Patient {pid} ---")
    print("CT  origin:", ct.GetOrigin(),  "size:", ct.GetSize(),  "spacing:", ct.GetSpacing(), "direction:", ct.GetDirection())
    print("PET origin:", pet.GetOrigin(), "size:", pet.GetSize(), "spacing:", pet.GetSpacing(), "direction:", pet.GetDirection())
    ct_extent  = tuple(s*sp for s, sp in zip(ct.GetSize(), ct.GetSpacing()))
    pet_extent = tuple(s*sp for s, sp in zip(pet.GetSize(), pet.GetSpacing()))
    print("CT  extent (mm):", ct_extent)
    print("PET extent (mm):", pet_extent)