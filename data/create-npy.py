import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import yaml
import SimpleITK as sitk
import pandas as pd

# Set quaternion threshold for nibabel if needed.
nib.Nifti1Header.quaternion_threshold = -1e-06
base_dir = "/labs/wanglab/projects/NeurIPS-Clean-T1T2"

def register_to_fixed(fixed_path, moving_path):
    # Read as ITK images
    fixed  = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    # Configure a simple rigid registration
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-6, numberOfIterations=100)
    registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY))
    registration_method.SetInterpolator(sitk.sitkLinear)

    transform = registration_method.Execute(fixed, moving)

    # Resample moving into fixed space
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampled = resampler.Execute(moving)

    # Convert back to numpy array, shape (X,Y,Z)
    arr = sitk.GetArrayFromImage(resampled)    # returns [Z,Y,X]
    return np.transpose(arr, (2,1,0))          # now [X,Y,Z]


# sits alongside your existing register_to_fixed()
def load_and_register(subject_path, modality, other_mod, dirs):
    """
    Load the 'fixed' volume for `modality` and 
    register the 'moving' volume for `other_modality` into its space.
    Returns a single numpy array in the fixed space.
    """
    fixed_path  = subject_path
    moving_path = Path(dirs[other_mod]) / subject_path.name
    if moving_path.exists():
        # sitk returns (X,Y,Z), nib ordering is (i,j,k) so transpose
        data = register_to_fixed(fixed_path, moving_path)
    else:
        data = nib.load(str(fixed_path)).get_fdata()
    return data

def construct_path(row, modality):
    return Path(base_dir) / row["dataset"] / "Preprocessed" / row["sub_id"] / row["ses_id"] / "anat"


def create_numpy_dataset_subject_splits(output_root, train_csv=None, val_csv=None, test_csv=None, slice_range=None, rotate=False):
    """
    Process NIfTI images and split them based on CSV-defined train/val/test subject lists.
    """
    output_root = Path(output_root)
    mapping = {}

    # Load CSVs and split validation from training
    train_df = pd.read_csv(train_csv) if train_csv else pd.DataFrame()
    val_df = train_df.iloc[:300].copy()
    train_df = train_df.iloc[300:].copy()
    test_df = pd.read_csv(test_csv) if test_csv else pd.DataFrame()

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    for split, df in splits.items():
        if df.empty:
            continue
        print(f"\nProcessing {len(df)} subjects for split '{split}'.")

        for modality in ["t1", "t2"]:
            split_dir = output_root / modality.lower() / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for _, row in df.iterrows():
                anat_path = construct_path(row, modality)
                scan_file = None
                if anat_path.exists():
                    for f in os.listdir(anat_path):
                        if f.endswith(".nii.gz") and modality.upper() in f and not f.endswith("_mask.nii.gz"):
                            scan_file = anat_path / f
                            break

                if not scan_file or not scan_file.exists():
                    print(f"[Warning] Missing file for {row['sub_id']} {modality}")
                    continue

                subj_id, ses_id = row["sub_id"], row["ses_id"]
                # print(f"  Processing subject: {subj_id}_{ses_id}_{modality}")
                img = nib.load(str(scan_file))
                data = np.squeeze(img.get_fdata())

                total_slices = int(data.shape[2])
                start, end = (total_slices // 2 - 1, total_slices // 2 + 2) if slice_range is None else (max(0, slice_range[0]), min(total_slices, slice_range[1]))
                # print(f"    Processing slices from index {start} to {end-1}")

                slice_files = []
                for i in range(start, end):
                    slice_data = data[:, :, i]
                    if rotate:
                        slice_data = np.rot90(slice_data, -1)
                    slice_data = slice_data - slice_data.min()
                    if slice_data.max() > 0:
                        slice_data = slice_data / (slice_data.max() + 1e-8)
                    slice_filename = split_dir / f"{subj_id}_{ses_id}_slice_{i}.npy"
                    np.save(str(slice_filename), slice_data.astype(np.float32))
                    slice_files.append(str(slice_filename))

                key = f"{modality.lower()}/{split}/{subj_id}_{ses_id}"
                mapping[key] = slice_files

    mapping_file = output_root / "subject_ids.yaml"
    with open(mapping_file, "w") as f:
        yaml.dump(mapping, f)

    print("\nDataset creation complete.")
    print(f"All subject slices are saved under: {output_root}")
    print(f"Subject mapping saved to: {mapping_file}")


if __name__ == "__main__":
    output_root = "datasets/nips"
    train_csv = "/home/xzhon54/xinliuz/datasets/train_split.csv"
    val_csv = None  # If separate CSV for val is not available
    test_csv = "/home/xzhon54/xinliuz/datasets/test_split.csv"

    create_numpy_dataset_subject_splits(output_root,
                                        train_csv=train_csv,
                                        val_csv=val_csv,
                                        test_csv=test_csv,
                                        slice_range=None,
                                        rotate=False)
