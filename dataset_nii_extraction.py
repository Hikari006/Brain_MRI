import os
import nibabel as nib
import matplotlib.pyplot as plt

# --- SETTINGS ---
# Folder containing your .nii or .nii.gz files
input_folder = 'datasets/AUTISM_YES'

# Folder where the output images will be saved
output_folder = 'datasets/AUTISM_YES_ext'
# --- END SETTINGS ---

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        print(f"Processing: {filename}")

        # Construct the full file path
        file_path = os.path.join(input_folder, filename)

        # Load the NIfTI file
        nii_img = nib.load(file_path)
        img_data = nii_img.get_fdata()

        # Get the base filename without the extension
        base_filename = filename.replace(".nii.gz", "").replace(".nii", "")

        # --- Save the middle slice from each axis ---

        # 1. Axial slice (view from the top)
        axial_slice_index = img_data.shape[2] // 2
        axial_slice = img_data[:, :, axial_slice_index]
        plt.imsave(
            os.path.join(output_folder, f"{base_filename}_axial.png"),
            axial_slice.T, # Transpose for correct orientation
            cmap='gray',
            origin='lower'
        )

        # 2. Coronal slice (view from the front)
        coronal_slice_index = img_data.shape[1] // 2
        coronal_slice = img_data[:, coronal_slice_index, :]
        plt.imsave(
            os.path.join(output_folder, f"{base_filename}_coronal.png"),
            coronal_slice.T,
            cmap='gray',
            origin='lower'
        )

        # 3. Sagittal slice (view from the side)
        sagittal_slice_index = img_data.shape[0] // 2
        sagittal_slice = img_data[sagittal_slice_index, :, :]
        plt.imsave(
            os.path.join(output_folder, f"{base_filename}_sagittal.png"),
            sagittal_slice.T,
            cmap='gray',
            origin='lower'
        )

print("--- Conversion Complete! ---")