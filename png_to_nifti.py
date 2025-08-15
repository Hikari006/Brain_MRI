import os
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import nibabel as nib

# -----------------------------
# Configuration
# -----------------------------
# Input folder with PNG slices (autism_no)
INPUT_DIR = 'datasets/autism_no_new'
# Output folder for NIfTI volumes
OUTPUT_DIR = 'datasets/AUTISM_NO_NII'
# If None, stack all slices into a single volume. Otherwise, split into chunks of this many slices per volume
SLICES_PER_VOLUME: Optional[int] = None  # e.g., 128
# Resize slices to a fixed size (width, height). If None, use size of the first slice.
RESIZE_TO: Optional[Tuple[int, int]] = None  # e.g., (256, 256)
# Save dtype and scaling
SAVE_DTYPE = np.float32  # keep normalized float data for 3D CNNs
SCALE_TO_UNIT = True     # if True, scale pixel values to [0, 1]


# -----------------------------
# Helpers
# -----------------------------
_digit_regex = re.compile(r'(\d+)')

def _numeric_key(filename: str):
    # Return a tuple to keep key types consistent: numeric-first, then name
    match = _digit_regex.search(filename)
    if match:
        return (0, int(match.group(1)), filename)
    return (1, 0, filename)


def _list_pngs_sorted(directory: str) -> List[str]:
    files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    files.sort(key=_numeric_key)
    return files


def _load_png_grayscale(path: str, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    img = Image.open(path).convert('L')
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img)
    return arr


def _stack_slices_to_volume(slice_paths: List[str], target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    first = _load_png_grayscale(slice_paths[0], target_size)
    height, width = first.shape
    depth = len(slice_paths)
    volume = np.empty((height, width, depth), dtype=np.float32)

    for idx, p in enumerate(slice_paths):
        arr = _load_png_grayscale(p, (width, height))
        volume[:, :, idx] = arr

    if SCALE_TO_UNIT:
        # Robust scaling per volume
        vmin = np.percentile(volume, 1.0)
        vmax = np.percentile(volume, 99.0)
        if vmax <= vmin:
            vmin, vmax = float(volume.min()), float(volume.max())
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)
        else:
            volume = np.zeros_like(volume)
    else:
        # Just cast to target dtype later
        pass

    return volume.astype(SAVE_DTYPE)


def convert_png_folder_to_nifti():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    png_files = _list_pngs_sorted(INPUT_DIR)
    if not png_files:
        print(f"No PNG files found in {INPUT_DIR}")
        return

    # Determine target size
    if RESIZE_TO is not None:
        target_size = RESIZE_TO
    else:
        probe = Image.open(os.path.join(INPUT_DIR, png_files[0])).convert('L')
        target_size = probe.size  # (width, height)

    # Build list of absolute paths
    paths = [os.path.join(INPUT_DIR, f) for f in png_files]

    if SLICES_PER_VOLUME is None:
        # Single volume
        volume = _stack_slices_to_volume(paths, target_size)
        affine = np.eye(4, dtype=np.float32)
        nii = nib.Nifti1Image(volume, affine)
        out_path = os.path.join(OUTPUT_DIR, 'autism_no_volume.nii.gz')
        nib.save(nii, out_path)
        print(f"Saved {out_path} with shape {volume.shape} and dtype {volume.dtype}")
    else:
        # Multiple volumes from chunks
        num_slices = len(paths)
        num_vols = int(np.ceil(num_slices / SLICES_PER_VOLUME))
        for vol_idx in range(num_vols):
            start = vol_idx * SLICES_PER_VOLUME
            end = min((vol_idx + 1) * SLICES_PER_VOLUME, num_slices)
            chunk = paths[start:end]
            if not chunk:
                continue
            volume = _stack_slices_to_volume(chunk, target_size)
            affine = np.eye(4, dtype=np.float32)
            nii = nib.Nifti1Image(volume, affine)
            out_path = os.path.join(OUTPUT_DIR, f'autism_no_volume_{vol_idx:03d}.nii.gz')
            nib.save(nii, out_path)
            print(f"Saved {out_path} with shape {volume.shape} and dtype {volume.dtype}")


if __name__ == '__main__':
    convert_png_folder_to_nifti()