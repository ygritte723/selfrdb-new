import os
import numpy as np
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm

# Change these paths
input_root = Path("/home/xzhon54/xinliuz/selfrdb-new/datasets/nips")
output_root = Path("/home/xzhon54/xinliuz/selfrdb-new/datasets/nips_resized_256")

# Make sure output directory exists
output_root.mkdir(parents=True, exist_ok=True)

# Resize all .npy files under the input directory
for npy_path in tqdm(list(input_root.rglob("*.npy"))):
    data = np.load(npy_path)

    if data.shape != (256, 256):  # skip if already correct size
        data_resized = resize(data, (256, 256), mode='constant', preserve_range=True, anti_aliasing=True)
    else:
        data_resized = data

    # Save to mirrored path under output_root
    relative_path = npy_path.relative_to(input_root)
    save_path = output_root / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), data_resized.astype(np.float32))

print(f"Resizing complete. All resized files saved to: {output_root}")