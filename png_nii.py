import os
import numpy as np
import nibabel as nib
from PIL import Image

# Paths
png_folder = "datasets/autism_no_new"   # Folder containing PNG files
output_folder = "datasets/autism_no_new_nii"
os.makedirs(output_folder, exist_ok=True)

# Loop through PNGs
for png_file in os.listdir(png_folder):
    if png_file.lower().endswith(".png"):
        # Load PNG as grayscale
        img = Image.open(os.path.join(png_folder, png_file)).convert('L')
        img_array = np.array(img)

        # Add a third dimension for slice depth (shape: X, Y, 1)
        img_array_3d = img_array[:, :, np.newaxis]

        # Create NIfTI object (identity affine since PNG lacks orientation metadata)
        nii_image = nib.Nifti1Image(img_array_3d, affine=np.eye(4))

        # Save with same base name
        nii_filename = os.path.splitext(png_file)[0] + ".nii.gz"
        nib.save(nii_image, os.path.join(output_folder, nii_filename))

        print(f"Saved: {nii_filename}")

print("✅ Conversion complete! Each PNG is now its own NIfTI file.")
