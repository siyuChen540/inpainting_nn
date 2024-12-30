import os
import numpy as np
from typing import Tuple

def crop_npy_files(
    input_dir: str,
    output_dir: str,
    patch_size: Tuple[int, int] = (256, 256),
    step_size: Tuple[int, int] = (200, 200)
):
    """
    Crop all .npy files in the given directory and save the cropped files to the output directory.

    Args:
        input_dir (str): The directory containing the original .npy files.
        output_dir (str): The directory to save the cropped files.
        patch_size (Tuple[int, int], optional): The size of the cropped patches. Default is (256, 256).
        step_size (Tuple[int, int], optional): The step size for moving the crop window. Default is (200, 200).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(input_dir, filename)
            data = np.load(filepath)
            height, width = data.shape

            # Calculate the starting position for cropping
            patches = []
            patch_rows = []
            patch_cols = []

            for i in range(0, height, step_size[0]):
                if i + patch_size[0] > height:
                    i = height - patch_size[0]
                for j in range(0, width, step_size[1]):
                    if j + patch_size[1] > width:
                        j = width - patch_size[1]
                    patch = data[i:i + patch_size[0], j:j + patch_size[1]]
                    patches.append(patch)
                    patch_rows.append(i)
                    patch_cols.append(j)
            
            # Remove duplicate starting positions for cropping
            unique_patches = {}
            for idx, (i, j) in enumerate(zip(patch_rows, patch_cols)):
                key = (i, j)
                if key not in unique_patches:
                    unique_patches[key] = patches[idx]

            # Save the cropped files
            for (i, j), patch in unique_patches.items():
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}_patch_r{i}_c{j}.npy"
                save_path = os.path.join(output_dir, new_filename)
                np.save(save_path, patch)

    print(f"Cropping completed. Cropped files saved in {output_dir}")



#########################
# stitching npy data
#########################
def stitching_npy(
    cropped_dir: str,
    output_dir: str,
    original_shape: Tuple[int, int] = (1440, 2040),
    patch_size: Tuple[int, int] = (256, 256),
    step_size: Tuple[int, int] = (200, 200)
):
    """
    Reconstruct the original .npy matrix from the cropped files.

    Args:
        cropped_dir (str): The directory containing the cropped .npy files.
        output_dir (str): The directory to save the reconstructed files.
        original_shape (Tuple[int, int], optional): The shape of the original matrix. Default is (1440, 2040).
        patch_size (Tuple[int, int], optional): The size of the cropped patches. Default is (256, 256).
        step_size (Tuple[int, int], optional): The step size for moving the crop window. Default is (200, 200).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # supose all files are named as '20020801_20020831_r{i}_c{j}.npy', and can be adjust to your own file name pattern
    files = [f for f in os.listdir(cropped_dir) if f.endswith('.npy')]
    # select all files with the same time, suppose all files are named as '20020801_20020831_r{i}_c{j}.npy'
    time_set = set()
    for f in files:
        parts = f.split('.')
        time_info = parts[1]  # example: '20020801_20020831'
        time_set.add(time_info)

    for time in time_set:
        # select files with the same time
        time_files = [f for f in files if f.split('.')[1] == time]
        stitched = np.zeros(original_shape)
        weight = np.zeros(original_shape)

        for f in time_files:
            parts = f.split('_')
            r_part = parts[-2]      # 'r{i}'
            c_part = parts[-1]      # 'c{j}.npy'
            i = int(r_part[1:])
            j = int(c_part[1:-4])   # remove 'c' and '.npy'

            patch = np.load(os.path.join(cropped_dir, f))
            stitched[i:i + patch_size[0], j:j + patch_size[1]] += patch
            weight[i:i + patch_size[0], j:j + patch_size[1]] += 1

        # avoid divide by zero
        stitched /= np.maximum(weight, 1)

        # Save the reconstructed file
        new_filename = f"reconstructed_{time}.npy"
        save_path = os.path.join(output_dir, new_filename)
        np.save(save_path, stitched)

    print(f"Reconstruction completed. Reconstructed files saved in {output_dir}")
