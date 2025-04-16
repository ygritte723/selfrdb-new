import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import yaml

# Set quaternion threshold for nibabel if needed.
nib.Nifti1Header.quaternion_threshold = -1e-06

def create_numpy_dataset_subject_splits(modality_dirs_dict, output_root, slice_range=None, rotate=False):
    """
    Process multiple NIfTI images in each modality directory and split the subjects (images)
    into training, validation, and testing groups using a 70:20:10 ratio. All slices from
    one image (subject) are saved into the corresponding split folder (without a subject subfolder),
    following the file naming convention expected by SelfRDB:
    
        <output_root>/
            <modality>/      
                train/
                    <subject_id>_slice_0.npy
                    <subject_id>_slice_1.npy
                    ...
                val/
                    <subject_id>_slice_0.npy
                    <subject_id>_slice_1.npy
                    ...
                test/
                    <subject_id>_slice_0.npy
                    <subject_id>_slice_1.npy
                    ...
    
    A YAML file at the root records a mapping from "<modality>/<split>/<subject_id>"
    to the list of slice file paths.
    
    Args:
         modality_dirs_dict (dict): Dictionary mapping modality names to directories containing NIfTI files.
         output_root (str): Root folder for the output dataset.
         slice_range (tuple or None): Tuple (start, end) specifying the range of slices to process along the third dimension.
                                      If None, all available slices will be processed.
         rotate (bool): If True, rotate each slice 90° clockwise.
    """
    output_root = Path(output_root)
    mapping = {}  # Will map "modality/split/subject_id" -> list of slice file paths

    for modality, mod_dir in modality_dirs_dict.items():
        mod_dir = Path(mod_dir)
        # List all NIfTI files (e.g., *.nii or *.nii.gz) in the modality directory.
        subject_files = sorted(mod_dir.glob("*.nii*"))
        if not subject_files:
            print(f"[Warning] No NIfTI files found in {mod_dir} for modality '{modality}'. Skipping modality.")
            continue

        num_subjects = len(subject_files)
        print(f"\nFound {num_subjects} subjects for modality '{modality}' in {mod_dir}")

        # Compute the number of subjects for each split using the 70:20:10 ratio.
        train_count = int(num_subjects * 0.7)
        val_count = int(num_subjects * 0.2)
        test_count = num_subjects - train_count - val_count

        # Print split counts.
        print(f"Splitting subjects into: {train_count} train, {val_count} val, {test_count} test.")

        # Split subject files (using sorted order; you can randomize if needed)
        train_subjects = subject_files[:train_count]
        val_subjects = subject_files[train_count:train_count + val_count]
        test_subjects = subject_files[train_count + val_count:]

        splits = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }

        for split, files in splits.items():
            print(f"\nProcessing {len(files)} subjects for modality '{modality}' in split '{split}'.")
            # Create output folder for the split (no subject subfolder here)
            split_dir = output_root / modality.lower() / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for file_path in files:
                subject_id = file_path.stem
                print(f"  Processing subject: {subject_id}")

                # Load the image file.
                img = nib.load(str(file_path))
                data = img.get_fdata()
                print(f"    Image shape: {data.shape}")

                total_slices = int(data.shape[2])
                if slice_range is None:
                    start, end = total_slices // 2 - 50, total_slices // 2 + 50
                else:
                    start = max(0, slice_range[0])
                    end = min(total_slices, slice_range[1])
                print(f"    Processing slices from index {start} to {end-1}")

                slice_files = []
                for i in tqdm(range(start, end), desc=f"Subject {subject_id} - {split}"):
                    slice_data = data[:, :, i]
                    if rotate:
                        slice_data = np.rot90(slice_data, -1)  # Rotate 90° clockwise if needed

                    # Scale the slice data to [0, 1].
                    slice_data = slice_data - slice_data.min()
                    if slice_data.max() > 0:
                        slice_data = slice_data / (slice_data.max() + 1e-8)

                    # Create a filename that includes the subject_id to avoid name collisions.
                    slice_filename = split_dir / f"{subject_id}_slice_{i}.npy"
                    np.save(str(slice_filename), slice_data.astype(np.float32))
                    slice_files.append(str(slice_filename))

                # Update the mapping for this subject.
                key = f"{modality.lower()}/{split}/{subject_id}"
                mapping[key] = slice_files

    # Save the mapping to a YAML file at the dataset root.
    mapping_file = output_root / "subject_ids.yaml"
    with open(mapping_file, "w") as f:
        yaml.dump(mapping, f)

    print("\nDataset creation complete.")
    print(f"All subject slices are saved under: {output_root}")
    print(f"Subject mapping saved to: {mapping_file}")


if __name__ == "__main__":
    # Dictionary mapping modality names to directories containing NIfTI images.
    modality_dirs_dict = {
        "t1": "/home/xzhon54/wangdata/data/M_CRIB/whole_brain/preprocessed_T1w",
        "t2": "/home/xzhon54/wangdata/data/M_CRIB/whole_brain/preprocessed_T2w"
    }
    output_root = "datasets/infant"  # Set your desired output directory
    
    # The slice_range parameter remains optional. If None, all slices are processed.
    create_numpy_dataset_subject_splits(modality_dirs_dict, output_root, slice_range=None, rotate=False)
