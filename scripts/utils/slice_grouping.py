import nibabel as nib
import numpy as np
import os

def load_mri_images(base_filename):
    flair = nib.load(f"{base_filename}_flair.nii.gz").get_fdata()
    t1 = nib.load(f"{base_filename}_t1.nii.gz").get_fdata()
    t1ce = nib.load(f"{base_filename}_t1ce.nii.gz").get_fdata()
    t2 = nib.load(f"{base_filename}_t2.nii.gz").get_fdata()
    seg = nib.load(f"{base_filename}_seg.nii.gz").get_fdata()
    return flair, t1, t1ce, t2, seg

def extract_slices(dataset, start, end):
    """
    Groups slices based on adjacent slices, in sets of 3, for each modality.
    Saves outputs in data/slices directory.
    """
    output_dir = os.path.join("data", "slices")
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        data, mask = dataset[idx]

        for i in range(start, end + 1):
            if i == 0 or i == data.shape[1] - 1:
                continue  # Skip boundary slices

            slice_group = np.stack([data[:, i - 1, :, :], data[:, i, :, :], data[:, i + 1, :, :]], axis=-1)
            seg_slice = mask[i, :, :]

            np.save(os.path.join(output_dir, f"slice_group_{idx}_{i}.npy"), slice_group)
            np.save(os.path.join(output_dir, f"seg_slice_{idx}_{i}.npy"), seg_slice)
