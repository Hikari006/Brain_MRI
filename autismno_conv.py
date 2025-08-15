import os
import cv2

src_dir = 'datasets/autism_no'
dst_dir = 'autism_no_new'
os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        img_path = os.path.join(src_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Create new filename with .png extension
            new_filename = os.path.splitext(filename)[0] + '.png'
            new_path = os.path.join(dst_dir, new_filename)
            cv2.imwrite(new_path, img)
            print(f"Converted {filename} to {new_filename} in {dst_dir}")
        else:
            print(f"Failed to read {filename}")

print("Conversion complete. All .png files are in autism_no_new.")