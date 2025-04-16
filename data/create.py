import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import yaml

nib.Nifti1Header.quaternion_threshold = -1e-06

def create_t1_numpy_dataset_category(t1_files_dict, output_root, modality='t1', split='test', slice_range=None, rotate=False):
    """
    Convert multiple T1 NIfTI images into a dataset organized by category.
    
    The created folder structure will be:
    
        <output_root>/
            <category>/       (e.g., fetal, atlas, infant, adult)
                <modality>/   (e.g., t1)
                    <split>/  (e.g., test)
                        <subject>/   (subject folder from file stem)
                            slice_0.npy
                            slice_1.npy
                            ...
        subject_ids.yaml  (mapping: "<category>/<subject>" -> list of slice file paths)
    
    Each slice is scaled to [0,1] and saved as a .npy file.
    
    Args:
        t1_files_dict (dict): Dictionary mapping category names to T1 NIfTI file paths.
        output_root (str): Root folder for the output dataset.
        modality (str): Modality name (default 't1').
        split (str): Data split (e.g., 'train', 'val', or 'test').
        slice_range (tuple or None): Tuple (start, end) specifying which slices to process.
                                     If None, all slices are processed.
        rotate (bool): If True, rotate each slice 90° clockwise.
    """
    output_root = Path(output_root)
    subject_mapping = {}  # Mapping: "category/subject" -> list of slice filenames

    for category, t1_file in t1_files_dict.items():
        # Create folder: <output_root>/<category>/<modality>/<split>/<subject>/
        subject_dir = output_root / category / modality / split 
        subject_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing category: {category}")
        img = nib.load(t1_file)
        data = img.get_fdata()
        print(f"  Image shape: {data.shape}")

        # Determine slice range along the third dimension (axial slices)
        if slice_range is None:
            start, end = 0, int(data.shape[2])
        else:
            start = max(0, slice_range[0])
            end = min(data.shape[2], slice_range[1])

        slice_filenames = []
        for i in tqdm(range(start, end), desc=f"Slices in {category}"):
            slice_data = data[:, :, i]
            if rotate:
                slice_data = np.rot90(slice_data, -1)  # Rotate 90° clockwise if needed

            # Scale the slice data to [0,1]
            slice_data = slice_data - slice_data.min()
            if slice_data.max() > 0:
                slice_data = slice_data / (slice_data.max() + 1e-8)

            # Save the slice as a .npy file (convert to float32)
            slice_filename = subject_dir / f"slice_{i}.npy"
            np.save(str(slice_filename), slice_data.astype(np.float32))
            slice_filenames.append(str(slice_filename))
        
        subject_mapping[f"{category}"] = slice_filenames

    # Save the subject mapping to a YAML file at the root of the dataset
    subject_ids_file = output_root / "subject_ids.yaml"
    with open(subject_ids_file, "w") as f:
        yaml.dump(subject_mapping, f)

    print(f"Dataset creation complete. Processed {len(subject_mapping)} subjects.")
    print(f"Slices are saved under: {output_root}")
    print(f"Subject mapping saved to: {subject_ids_file}")

if __name__ == "__main__":
    # Define the mapping from category to T1 file path.
    # Adjust the paths according to your data.
    t1_files_dict = {
        # "fetal": '/home/xzhon54/xinliuz/nips_data/sub-CC00856XX15_ses-3530_desc-restore_T2w.nii.gz',
        # "atlas": '/home/xzhon54/xinliuz/nips_data/STA21.nii.gz',
        # "infant": '/home/xzhon54/xinliuz/nips_data/M-CRIB_P01_T1.nii',
        "adult": '/home/xzhon54/xinliuz/nips_data/1101_3.nii'
    }
    output_root = "datasets/nips"  # Set your desired output directory
    # Optionally, specify a slice range (e.g. slice_range=(27, 127)) or use None to process all slices.
    create_t1_numpy_dataset_category(t1_files_dict, output_root, modality='t1', split='test', slice_range=None, rotate=False)