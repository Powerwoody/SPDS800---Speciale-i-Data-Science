# Imports:
import os
import h5py
import numpy as np
import pandas as pd
import re
from datetime import datetime
from wrapper_main import Wrapper 
import tqdm
import shutil
from ruamel.yaml import YAML
        
class LightniningPoseWrapper(Wrapper):
    """
    Wrapper for converting DeepPoseKit HDF5 dataset into LightningPose project
    """
    def __init__(self, output_dir):
        super().__init__(output_dir)
        
        # Checking if the output directory exsists
        if not os.path.isdir(self.output_dir):
            raise NotADirectoryError(f"Output directory does not exist: {self.output_dir}")
    
    @staticmethod
    def edit_yaml_config_lp(
        config_filepath: str,
        config_dict: dict,
        output_dir: str = None,
        config_name: str = "config_updated.yaml",
        overwrite: bool = False
    ) -> None:
        """
        Edits a YAML config file (including nested keys), preserving formatting and null values.
        Writes to disk as `.yaml`.
        """
        config_name = config_name + ".yaml"
        
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.default_flow_style = False

        with open(config_filepath, "r") as f:
            config = yaml.load(f)

        def recursive_update(cfg, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                    recursive_update(cfg[key], value)
                else:
                    cfg[key] = value

        # Apply changes
        recursive_update(config, config_dict)

        # Determine output path
        if overwrite:
            output_path = config_filepath
        else:
            if output_dir is None:
                output_dir = os.getcwd()
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, config_name)

            # Prevent accidental overwrite
            if os.path.abspath(output_path) == os.path.abspath(config_filepath):
                raise ValueError("Output path is the same as input but overwrite=False.")

        # Write to YAML
        with open(output_path, "w") as f:
            yaml.dump(config, f)

        
    def h5_to_lp_project(
        self, 
        h5_file_path:str, 
        skeleton_csv_path:str,
        project_name:str,
        video_path:str,
        default_config_path:str
    )->None:

        # Input validation:
        if not os.path.isfile(h5_file_path):
            raise FileNotFoundError(f"h5_file_path does not exist: {h5_file_path}")
        if not os.path.isfile(skeleton_csv_path):
            raise FileNotFoundError(f"skeleton_csv_path does not exist: {skeleton_csv_path}")
        
        # Getting todays day:
        todays_date = datetime.today().strftime('%Y-%m-%d')
        
        # Creating a LP project folder at the output dir:
        print("Building Lightning Pose project...")
        project_name = project_name + "-" + todays_date
        
        project_dir = os.path.join(self.output_dir, project_name)
        video_dir = os.path.join(project_dir, "videos")
        labeled_data_dir = os.path.join(project_dir, "labeled-data")
        
        os.makedirs(project_dir)
        os.makedirs(os.path.join(project_dir, video_dir), exist_ok=True)
        os.makedirs(os.path.join(project_dir, labeled_data_dir), exist_ok=True)
        
        # Copying video to LP project folder:
        shutil.copy2(video_path, video_dir)
        
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
        
        # Extracting all frames and adding them to the labeled-data dir inside of the LP project dir:
        print("Writing frames to LP project...")
        LightniningPoseWrapper.write_img_from_h5(labeled_data_dir, images)
            
        # Getting the keypoint names:
        skeleton_df = pd.read_csv("/Users/marcusschou/Desktop/SDU/data_science/kurser_4/Speciale_Data_Science/data/skeletons/natskeleton_noposs.csv")
        keypoint_names = skeleton_df["name"].to_list()
        num_keypoints = len(keypoint_names)
        
        num_frames, width, height, channels = images.shape
        
        config_dict = {
            "data": {
                "data_dir": labeled_data_dir,
                "video_dir": video_dir,
                "num_keypoints": num_keypoints,
                "keypoint_names": keypoint_names,
                "mirrored_column_matches": None,
                "columns_for_singleview_pca": None,
                "image_resize_dims": {
                    "height": height,
                    "width": width
                }
                
            }
        }

        LightniningPoseWrapper.edit_yaml_config_lp(
            config_filepath=default_config_path,
            config_dict=config_dict,
            config_name= project_name,
            output_dir=project_dir
        )
        
        # Creating CSV file:
        
        # Making a list with all the image paths in the dlc project folder:
        image_files = sorted(
        [f for f in os.listdir(labeled_data_dir) if f.lower().endswith('.png')],
        key=lambda x: int(re.findall(r'\d+', x)[0])
        )
        
        image_files = [f"labeled-data/" + f for f in image_files]
        
        num_cols = len(annotations[1])
        header_row = ["scorer"] + ["experimenter"] * (num_cols) * 2
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
        output_csv = project_dir + "/" + f"CollectedData.csv"
        df.to_csv(output_csv, header=False, index=False)
        
        print(f"LP project is ready at {self.output_dir}")
        