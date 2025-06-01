import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import sleap_io as sio 
from wrapper_main import *
        
class SleapWrapper(Wrapper):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        
    def convert_to_slp(self, h5_file_path:str, skeleton_csv_path:str)->None:
        # Input validation
        if not os.path.isfile(h5_file_path):
            raise FileNotFoundError(f"h5_file_path does not exist: {h5_file_path}")
        if not os.path.isfile(skeleton_csv_path):
            raise FileNotFoundError(f"skeleton_csv_path does not exist: {skeleton_csv_path}")

        # Creating output filepaths:
        labels_filepath = os.path.join(self.output_dir, os.path.basename(h5_file_path).replace(".h5", ".slp"))
        video_filepath = os.path.join(self.output_dir, os.path.basename(h5_file_path).replace(".h5", ".mp4"))

        # Reading the .h5 file:
        try:
            with h5py.File(h5_file_path, "r") as f:
                annotations = f["annotations"][:] 
                images = f["images"][:] 
                annotated = f["annotated"][:] 
        except Exception as e:
            raise RuntimeError(f"Error while reading .h5 file: {e}")

        # Preprocessing: remove unannotated frames
        try:
            annotations, images = SleapWrapper.preprocessing_frames(annotations, images, annotated)
        except Exception as e:
            print(f"Error while preprocessing .h5 file: {e}")
        
        # Reading the skeleton CSV and building edges:
        skeleton_data = pd.read_csv(skeleton_csv_path, header=0)
        edges = []
        for name, parent, swap in skeleton_data.itertuples(index=False, name=None):
            if pd.notna(parent): 
                edges.append((parent, name))

        # Building the sleap skeleton object:
        skeleton = sio.Skeleton(
            nodes=skeleton_data["name"].tolist(),
            edges=edges
        )

        # Creating a .mp4 video from images only if it doesn't already exist
        if not os.path.exists(video_filepath):
            try:
                with sio.VideoWriter(video_filepath) as writer:
                    for frame in tqdm.tqdm(images):
                        writer(frame)
            except Exception as e:
                raise RuntimeError(f"Error while writing video: {e}")
            print(f"Video saved to: {video_filepath}")
        else:
            print(f"Video file already exists: {video_filepath}")

        video = sio.load_video(video_filepath)

        # Building sleap LabeledFrame objects for each frame:
        lfs = []
        for frame_idx in tqdm.tqdm(range(annotations.shape[0])):
            pts = annotations[frame_idx]
            instance = sio.Instance.from_numpy(points=pts, skeleton=skeleton)
            lf = sio.LabeledFrame(
                video=video,
                frame_idx=frame_idx,
                instances=[instance],
            )
            lfs.append(lf)

        # Building a sleap Labels object:
        labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=lfs)

        # Saving the labeled file to output dir:
        labels.save(labels_filepath, embed=True)
        print(f"SLEAP labels file saved to: {labels_filepath}")
        
    @staticmethod
    def split_indicies(slp_filepath:str, train_split_percentage:float=0.8, test_split_percentage:float=0.2, seed:int=42):
            # Check if file exists
            if not os.path.isfile(slp_filepath):
                raise FileNotFoundError(f"h5_file_path does not exist: {slp_filepath}")
            
            # Ensure the split percentages add up to 1.0
            if not np.isclose(train_split_percentage + test_split_percentage, 1.0):
                return "Split percentage has to add up to 1.0"
            
            # Open the h5 file and read the frames dataset
            with h5py.File(slp_filepath, "r") as f:
                frames = f["frames"][:]  # e.g., shape (num_frames, )
            
            num_frames = len(frames)
            # Generate a list of all indices
            indices = np.arange(num_frames)
            
            # Set the seed to ensure reproducibility
            np.random.seed(seed)
            np.random.shuffle(indices)  # Shuffle the indices randomly
            
            # Determine the number of training frames
            train_size = int(train_split_percentage * num_frames)
            
            # Split indices into training and testing
            train_split_indices = indices[:train_size].tolist()
            test_split_indices = indices[train_size:].tolist()
            
            return train_split_indices, test_split_indices
        
        
        
        
    
    
    
    
             
