# Imports:
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import re
from datetime import datetime
import deeplabcut as dlc 
from deeplabcut.utils.auxiliaryfunctions import edit_config 
from wrapper_main import Wrapper 
import tqdm
        
class DeepLabCutWrapper(Wrapper):
    """
    Wrapper for converting DeepPoseKit HDF5 dataset into DeepLabCut dataset
    """
    def __init__(self, output_dir):
        super().__init__(output_dir)
        
        if not os.path.isdir(self.output_dir):
            raise NotADirectoryError(f"Output directory does not exist: {self.output_dir}")
        
    def h5_to_dlc_project(
        self, 
        h5_file_path:str, 
        skeleton_csv_path:str,
        project_name:str,
        experimenter:str,
        video_paths:list[str],
    )->None:

        # Input validation:
        if not os.path.isfile(h5_file_path):
            raise FileNotFoundError(f"h5_file_path does not exist: {h5_file_path}")
        if not os.path.isfile(skeleton_csv_path):
            raise FileNotFoundError(f"skeleton_csv_path does not exist: {skeleton_csv_path}")
        
        # Creating a new DLC project folder at the output dir:
        dlc.create_new_project(
            project=project_name,
            experimenter=experimenter,
            videos=video_paths,
            working_directory=self.output_dir
        )
                
        # Getting todays date:
        todays_date = datetime.today().strftime('%Y-%m-%d')
        # Getting the project dir for the created dlc project:
        project_dir = self.output_dir + project_name + "-" + experimenter + "-" + todays_date
        # Getting the path for the created project config file:
        config_path = os.path.join(project_dir, "config.yaml")
        video_filename = os.path.basename(video_paths[0]).replace(".mp4", "")
        # Gettin the path for the created project labeled-data dir:        
        labeled_data_dir = os.path.join(project_dir, "labeled-data", video_filename)
        
        # Informative print statements:
        print(50 * "#")
        print(f"Project dir: {project_dir}")
        print(f"Config filepath: {config_path}")
        print(f"labeled-data dir: {labeled_data_dir}")
        print(50 * "#")
        
        # Reading .h5 file:
        with h5py.File(h5_file_path, "r") as f:
            images = f["images"][:]
            annotations = f["annotations"][:]
            annotated = f["annotated"][:]
            
        # Preprocessing frames:   
        annotations, images = Wrapper.preprocessing_frames(
            annotations=annotations,
            images=images,
            annotated=annotated
        )
        
        # Extracting all frames and adding them to the labeled-data dir inside of the dlc project dir:
        print("Writing frames to DLC project")
        DeepLabCutWrapper.write_img_from_h5(labeled_data_dir, images)
            
        # Getting the keypoint names:
        skeleton_df = pd.read_csv("/Users/marcusschou/Desktop/SDU/data_science/kurser_4/Speciale_Data_Science/data/skeletons/natskeleton_noposs.csv")
        keypoint_names = skeleton_df["name"].to_list()
        skeleton_list = []
        
        # Making the skeleton structure
        print("Adding the skeleton to the config.yaml file")
        for index, row in tqdm.tqdm(skeleton_df.iterrows()):
            if pd.notna(row["parent"]):
                edge = [row["parent"], row["name"]]
                skeleton_list.append(edge)
        
        # Adding the keypoint names and skeleton structure to the config.yaml file in the project folder:
        edit_config(config_path, {"bodyparts": keypoint_names})
        edit_config(config_path, {"skeleton": skeleton_list}) # Just used for vis not for augmentation!
        edit_config(config_path, {"skeleton_color": "red"}) 
        
        # Adding the number of frames to pick to the config file just so that its added:
        edit_config(config_path, {"numframes2pick": images[0]})
        
        # Creating CSV file:
        
        # Making a list with all the image paths in the dlc project folder:
        image_files = sorted(
        [f for f in os.listdir(labeled_data_dir) if f.lower().endswith('.png')],
        key=lambda x: int(re.findall(r'\d+', x)[0])
        )
        
        image_files = [f"labeled-data/{video_filename}/" + f for f in image_files]
        
        num_cols = len(annotations[1])
        header_row = ["scorer"] + [f"{experimenter}"] * (num_cols) * 2
        bodyparts_row = bodyparts_row = ["bodyparts"] + [kp_name for kp_name in keypoint_names for _ in ("x", "y")]
        coords_row = ["coords"] + (["x"] + ["y"]) * (num_cols)

        csv_data = [header_row, bodyparts_row, coords_row]

        print("Creating the labeled-data.csv for the DLC project")
        for i, img_file in tqdm.tqdm(enumerate(image_files)):
            row = [img_file]
            for j in range(num_cols):
                x, y = annotations[i, j, 0], annotations[i, j, 1]
                row.extend([x if not np.isnan(x) else "", y if not np.isnan(y) else ""])
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        output_csv = labeled_data_dir + "/" + f"CollectedData_{experimenter}.csv"
        df.to_csv(output_csv, header=False, index=False)
        
        dlc.convertcsv2h5(config_path, userfeedback=False) # Making the dlc .h5 from the csv
        # dlc.check_labels(config_path) # Adding labeles to frames (For sanity check)
        print(f"DLC project is ready at {self.output_dir}")
    
    @staticmethod
    def split_indicies(collected_data_csv_path, train_split_percentage:float=0.8, test_split_percentage:float=0.2, seed:int=42):
            # Check if file exists
            if not os.path.isfile(collected_data_csv_path):
                raise FileNotFoundError(f"h5_file_path does not exist: {collected_data_csv_path}")
            
            # Ensure the split percentages add up to 1.0
            if not np.isclose(train_split_percentage + test_split_percentage, 1.0):
                return "Split percentage has to add up to 1.0"
            
            # Open the h5 file and read the frames dataset
            df = pd.read_csv(collected_data_csv_path)
            num_frames = len(df[2:])
            
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