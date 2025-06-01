import numpy as np
import h5py
import os
import cv2
import tqdm


class Wrapper:
    def __init__(self, output_dir:str) -> None:
        self.output_dir = output_dir
        
        # Input validations:
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
    @staticmethod
    def preprocessing_frames(annotations:np.ndarray, images:np.ndarray, annotated:np.ndarray) ->tuple:
        """Preprocesses frames by setting (x, y) coordinates to NaN when keypoints are (-1, -1) and
        filters out frames where all keypoints are marked as False in the annotated dataset from the HDF5 file.

        Args:
            annotations (np.ndarray): Array of (x, y) coordinates for each keypoint in each frame from the .h5 file.
            images (np.ndarray): Array of frames from the .h5 file.
            annotated (np.ndarray): Boolean array indicating whether each keypoint is annotated for each frame.

        Returns:
            tuple: A tuple containing the filtered annotations and filtered images.
        """
        
        # Identifing frames where all values in the annotated row are False:
        unannotated_frames = np.all(annotated == False, axis=1)
        frames_to_keep = ~unannotated_frames

        print(f"Total frames: {len(annotated)}")
        print(f"Frames marked for removal: {np.sum(unannotated_frames)}")
        print(f"Frames kept: {np.sum(frames_to_keep)}")

        # Applying mask to filter out completely unannotated frames:
        filtered_annotations = annotations[frames_to_keep]
        filtered_images = images[frames_to_keep]

        # Ensuring missing keypoints (-1, -1) are replaced with NaN:
        filtered_annotations[(filtered_annotations[:, :, 0] == -1) & (filtered_annotations[:, :, 1] == -1)] = np.nan

        print(f"New Annotations shape: {filtered_annotations.shape}")
        print(f"New Images shape: {filtered_images.shape}")

        return filtered_annotations, filtered_images
    
    @staticmethod
    def inspect_h5_file(h5_file_path:str)->None:
        """_summary_

        Args:
            h5_file_path (str): _description_
        """
        with h5py.File(h5_file_path, 'r') as f:
            for key in f.keys():
                h5_data = f[key]
                print(f"Key: {key}")
                
                if isinstance(h5_data, h5py.Dataset):
                    first_value = h5_data[0] if h5_data.shape[0] > 0 else None
                    print(f"First value: {first_value}\n")
                else:
                    print("Not a dataset...\n")
    
    @staticmethod        
    def write_img_from_h5(h5_file_path:str, output_dir:str)->None:
        """Writes images from .h5 files to an specific output directory

        Args:
            h5_file_path (str): Path to h5 file
            output_dir (str): Output directory where the images will be written to
        """
        try:
            with h5py.File(h5_file_path, "r") as f:
                images = f["images"][:]
        except:
            KeyError(f"No key named 'images' in the {os.path.basename(h5_file_path)}")
         
        
            for i, frame in tqdm.tqdm(enumerate(images)):
                if frame.ndim == 3 and frame.shape[-1] == 1:
                    frame = frame.squeeze(-1)
                
            frame_filename = os.path.join(output_dir, f"img{i:03d}.png")
            cv2.imwrite(frame_filename, frame)  
    
