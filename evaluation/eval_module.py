# Imports:
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pandas.errors import ParserError
import h5py
import os
import re
from pathlib import Path

# __________________________________________Test set evalutaion:__________________________________________

def plot_loss_curves(toolbox:str, model_name:str, trained_model_path:str, trained_centroid_model_path:str = None):
    """This function plots the loss curves for trained models."""
    
    try:
        trainlogs_csv = pd.read_csv(trained_model_path)
    except ParserError:
        print(f"The filepath: {os.path.basename(trained_model_path)} is not a valid CSV file.")
        return

    if trained_centroid_model_path is not None:
        try:
            trainlogs_csv_centroid = pd.read_csv(trained_centroid_model_path)
        except ParserError:
            print(f"The filepath: {os.path.basename(trained_centroid_model_path)} is not a valid CSV file.")
            return

    if str(toolbox) == "slp":
        if trained_centroid_model_path is not None:
            required_columns = {"epoch", "loss", "val_loss"}
            if not required_columns.issubset(trainlogs_csv.columns) or not required_columns.issubset(trainlogs_csv_centroid.columns):
                raise ValueError("One or both CSV files are missing required columns: epoch, loss, val_loss.")

            # Epochs:
            epochs = trainlogs_csv["epoch"].tolist()
            epochs_val = trainlogs_csv["epoch"].tolist()
            epochs_centroid = trainlogs_csv_centroid["epoch"].tolist()
            epochs_centroid_val = trainlogs_csv_centroid["epoch"].tolist()

            # Loss values main:
            train_loss = trainlogs_csv["loss"].tolist()
            val_loss = trainlogs_csv["val_loss"].tolist()
            
            # Loss values centroid:
            train_loss_centroid = trainlogs_csv_centroid["loss"].tolist()
            val_loss_centroid = trainlogs_csv_centroid["val_loss"].tolist()
        else:
            required_columns = {"epoch", "loss", "val_loss"}
            if not required_columns.issubset(trainlogs_csv.columns):
                raise ValueError("CSV file is missing required columns: epoch, loss, val_loss.")
            
            epochs = trainlogs_csv["epoch"].tolist()
            epochs_val = trainlogs_csv["epoch"].tolist()
            train_loss = trainlogs_csv["loss"].tolist()
            val_loss = trainlogs_csv["val_loss"].tolist()
    elif str(toolbox) == "dlc":
        if trained_centroid_model_path is not None:
            required_columns = {"step", "losses/train.total_loss", "losses/eval.total_loss"}
            if not required_columns.issubset(trainlogs_csv.columns) or not required_columns.issubset(trainlogs_csv_centroid.columns):
                raise ValueError("One or both CSV files are missing required columns: epoch, loss, val_loss.")
            
            epochs = trainlogs_csv["step"].tolist()
            epochs_val = [i for i in range(10, len(epochs)+1, 10)]
            epochs_centroid = trainlogs_csv_centroid["step"].tolist()
            epochs_centroid_val = [i for i in range(10, len(epochs_centroid)+1, 10)]
            
            # Loss values main:
            train_loss = trainlogs_csv["losses/train.total_loss"].tolist()
            val_loss = trainlogs_csv["losses/eval.total_loss"].tolist()
            val_loss = [x for x in val_loss if not np.isnan(x)]
            
            # Loss values centroid:
            train_loss_centroid = trainlogs_csv_centroid["losses/train.total_loss"].tolist()
            val_loss_centroid = trainlogs_csv_centroid["losses/eval.total_loss"].tolist()
            val_loss_centroid = [x for x in val_loss if not np.isnan(x)]
        else:
            required_columns = {"step", "losses/train.total_loss", "losses/eval.total_loss"}
            if not required_columns.issubset(trainlogs_csv.columns):
                raise ValueError("CSV file is missing required columns: step, losses/train.total_loss, losses/eval.total_loss.")
            
            epochs = trainlogs_csv["step"].tolist()
            epochs_val = [i for i in range(10, len(epochs)+1, 10)]
            
            train_loss = trainlogs_csv["losses/train.total_loss"].tolist()
            val_loss = trainlogs_csv["losses/eval.total_loss"].tolist()
            val_loss = [x for x in val_loss if not np.isnan(x)]
    elif str(toolbox) == "lp":
        required_columns = {"train_supervised_loss", "epoch", "val_supervised_loss"}
        if not required_columns.issubset(trainlogs_csv.columns):
            raise ValueError("One or both CSV files are missing required columns: epoch, loss, val_loss.")
        
        train_loss = trainlogs_csv.groupby("epoch")["train_supervised_loss"].mean().dropna()
        val_loss = trainlogs_csv.groupby("epoch")["val_supervised_loss"].mean().dropna()

        epochs_val = val_loss.index.tolist()
        epochs = train_loss.index.tolist()
        
        train_loss = train_loss.tolist()
        val_loss = val_loss.tolist()
    else:
        return f"This: f{toolbox} is not valid toolbox (valid toolboxes: slp, dlc, lp)"
    
    # Plotting:
    # Plotting main model
    plt.figure(figsize=(5, 5))
    plt.plot(epochs, train_loss, label="Training Loss")
    if val_loss != []:
        plt.plot(epochs_val, val_loss, label="Validation Loss")
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.title(f"{model_name} (Main)", fontsize=15)
    plt.legend()
    plt.show()

    # Plotting centroid model if available
    if trained_centroid_model_path is not None:
        plt.figure(figsize=(5, 5))
        plt.plot(epochs_centroid, train_loss_centroid, label="Training Loss Centroid")
        if val_loss_centroid != []:
            plt.plot(epochs_centroid_val, val_loss_centroid, label="Validation Loss Centroid")
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.title(f"{model_name} (Centroid)", fontsize=15)
        plt.legend()
        plt.show()
# DONE     
def extract_test_pred_slp(pred_h5_path:str, gt_h5_path:str)->tuple[np.ndarray, np.ndarray, list]:
    NUM_KEYPOINTS = 20

    try:
        with h5py.File(pred_h5_path, "r") as f:
            if "pred_points" in f:
                pred_points = f["pred_points"][:]
                frames = f["frames"][:]
            else:
                raise KeyError(f"'pred_points' key not found in {pred_h5_path}!")
    except FileNotFoundError:
        raise FileNotFoundError(f"{pred_h5_path} does not exist!")

    try:
        with h5py.File(gt_h5_path, "r") as f:
            if "points" in f:
                gt_points = f["points"][:]
            else:
                raise KeyError(f"'points' key not found in {gt_h5_path}!")
    except FileNotFoundError:
        raise FileNotFoundError(f"{gt_h5_path} does not exist!")

    val_frames_idx = frames["frame_idx"].tolist()

    num_test_frames_pred = int(len(pred_points) / NUM_KEYPOINTS)
    num_test_frames_gt = int(len(gt_points) / NUM_KEYPOINTS)

    if num_test_frames_pred != num_test_frames_gt:
        raise ValueError("Predictions do not match ground truths in number of frames!")

    pred_points = np.array([(x, y, conf) for (x, y, _, _, conf) in pred_points])
    pred_points = pred_points.reshape(num_test_frames_pred, NUM_KEYPOINTS, 3)
    
    gt_points = np.array([(x, y) for (x, y, _, _) in gt_points])
    gt_points = gt_points.reshape(num_test_frames_gt, NUM_KEYPOINTS, 2)

    return pred_points, gt_points, val_frames_idx
# DONE
def extract_test_pred_dlc(pred_csv_path:str, collected_data_csv_path:str)->tuple[np.ndarray, np.ndarray, list]:
    # Read CSV files
    
    test_indices = [821, 705, 1101, 877, 895, 913, 1210, 152, 1001, 1156, 509, 111, 226, 1008, 851, 103, 421, 419, 967, 624, 586, 1111, 919, 672, 119, 53, 151, 403, 1200, 207, 1145, 658, 1107, 232, 8, 159, 36, 452, 651, 253, 303, 896, 1186, 880, 571, 1011, 623, 345, 1041, 1036, 262, 610, 297, 989, 414, 150, 472, 640, 1201, 1140, 550, 928, 488, 1091, 146, 402, 954, 1135, 659, 463, 186, 608, 143, 751, 981, 197, 883, 279, 293, 1099, 400, 122, 202, 835, 246, 1150, 384, 854, 219, 641, 1, 112, 698, 951, 1089, 441, 663, 1106, 317, 648, 709, 972, 627, 632, 1182, 1117, 795, 645, 1012, 556, 681, 577, 524, 1059, 540, 748, 1120, 484, 95, 1020, 563, 742, 863, 891, 1038, 206, 392, 794, 870, 397, 766, 1110, 1028, 642, 612, 960, 725, 683, 98, 804, 406, 502, 1071, 1056, 929, 779, 200, 134, 1051, 40, 1017, 230, 378, 288, 418, 391, 592, 1086, 647, 520, 64, 14, 1064, 492, 379, 187, 763, 216, 791, 1076, 878, 337, 719, 295, 1016, 455, 815, 269, 995, 201, 161, 729, 401, 702, 1129, 565, 1021, 1025, 1104, 205, 34, 775, 508, 1195, 91, 897, 564, 776, 241, 13, 315, 600, 387, 166, 840, 20, 646, 1154, 831, 562, 1181, 686, 957, 189, 975, 699, 510, 1082, 474, 856, 747, 252, 21, 459, 1184, 276, 955, 385, 805, 343, 769, 130, 871, 1123, 87, 330, 466, 121, 1044, 1095, 1130, 860, 1126]
    df_pred = pd.read_csv(pred_csv_path, header=2)
    df_train = pd.read_csv(collected_data_csv_path, header=2)

    # Extract frame numbers from the "coords" field
    df_train["coords"] = df_train["coords"].apply(lambda x: int(x.split("img")[1].split(".")[0].lstrip("0") or "0"))

    # Set 'coords' as index temporarily to use .loc[]
    df_train_indexed = df_train.set_index("coords")

    # Select rows using test_indices in exact order
    df_test_ordered = df_train_indexed.loc[test_indices].reset_index()
    df_test_ordered = df_test_ordered.drop("coords", axis=1)
    
    df_pred = df_pred.drop("coords", axis=1)
    
    # Convert to floats, preserve NaN
    df_pred = df_pred.apply(pd.to_numeric, errors='coerce')
    df_gt = df_test_ordered.apply(pd.to_numeric, errors='coerce')
    
    # Create NumPy arrays
    n_keypoints = 20
    n_frames = len(df_pred)
    pred = df_pred.to_numpy(dtype=np.float32).reshape(n_frames, n_keypoints, 3)
    gt = df_gt.to_numpy(dtype=np.float32).reshape(n_frames, n_keypoints, 2)
    
    return pred, gt, test_indices
# DONE  
def extract_test_pred_lp_bu(pred_csv_path:str, collected_data_gt_csv_path:str)->tuple[np.ndarray, np.ndarray, list]:
    # Load predictions
    df_pred = pd.read_csv(pred_csv_path)
    df_pred = df_pred[df_pred["set"] == "validation"]
    
    # Process scorer
    df_pred["scorer"] = df_pred["scorer"].apply(lambda x: int(x.split("img")[1].split(".")[0].lstrip("0") or "0"))
    frame_idx = df_pred["scorer"].tolist()
    df_pred = df_pred.drop("set", axis=1)
    
    # Load ground truth
    df_gt = pd.read_csv(collected_data_gt_csv_path)
    df_gt = df_gt[2:]
    df_gt["scorer"] = df_gt["scorer"].apply(lambda x: int(re.search(r'\d+', x).group()))
    df_gt = df_gt[df_gt["scorer"].isin(frame_idx)]
    
    # Drop 'scorer' column
    df_pred = df_pred.drop("scorer", axis=1)
    df_gt = df_gt.drop("scorer", axis=1)
    
    # Check column counts
    expected_pred_cols = 60  # 20 keypoints * 3 (x, y, confidence)
    if len(df_pred.columns) != expected_pred_cols:
        # Select first 60 columns (or adjust based on inspection)
        df_pred = df_pred.iloc[:, :expected_pred_cols]
    
    expected_gt_cols = 40  # 20 keypoints * 2 (x, y)
    if len(df_gt.columns) != expected_gt_cols:
        df_gt = df_gt.iloc[:, :expected_gt_cols]
    
    # Convert to floats, preserve NaN
    df_pred = df_pred.apply(pd.to_numeric, errors='coerce')
    df_gt = df_gt.apply(pd.to_numeric, errors='coerce')
    
    # Create NumPy arrays
    n_keypoints = 20
    n_frames = len(df_pred)
    pred = df_pred.to_numpy(dtype=np.float32).reshape(n_frames, n_keypoints, 3)
    gt = df_gt.to_numpy(dtype=np.float32).reshape(n_frames, n_keypoints, 2)
    
    return pred, gt, frame_idx
# DONE
def extract_source_and_frame(path:str)->tuple[str, int]:
    parts = path.split(os.sep)
    video_name = parts[-3]  # e.g., 'Windtunnel_Prey_Capture_Approach_...'
    frame_id = int(os.path.basename(path).replace(".png", ""))
    return video_name, frame_id
# DONE
def extract_test_pred_lp_context(pred_csv_path:str, collected_data_gt_csv_path:str, mapping_csv_path:str)->tuple[np.ndarray, np.ndarray, list]:
    mapping_df = pd.read_csv(mapping_csv_path)
    mapping_df["org_img_idx"] = range(len(mapping_df))
    mapping_df["matlabFrameIdx"] = mapping_df["matlabFrameIdx"].astype(int)

    # --- Load and process predictions ---
    df_pred = pd.read_csv(pred_csv_path, header=2)
    df_pred = df_pred[df_pred["Unnamed: 61"] == "validation"]

    df_pred[["sourceVideo", "matlabFrameIdx"]] = df_pred["coords"].apply(
        lambda p: pd.Series(extract_source_and_frame(p))
    )
    df_pred["matlabFrameIdx"] = df_pred["matlabFrameIdx"].astype(int)

    # --- Map to original indices ---
    df_pred_merged = pd.merge(
        df_pred,
        mapping_df[["sourceVideo", "matlabFrameIdx", "org_img_idx"]],
        on=["sourceVideo", "matlabFrameIdx"],
        how="left"
    )

    mapped_indices = df_pred_merged["org_img_idx"].tolist()

    # --- Load and process ground truth ---
    df_gt = pd.read_csv(collected_data_gt_csv_path, header=2)
    df_gt[["sourceVideo", "matlabFrameIdx"]] = df_gt["coords"].apply(
        lambda p: pd.Series(extract_source_and_frame(p))
    )
    df_gt["matlabFrameIdx"] = df_gt["matlabFrameIdx"].astype(int)

    # Match GT to pred using both sourceVideo and frame index
    df_gt = pd.merge(
        df_gt,
        df_pred[["sourceVideo", "matlabFrameIdx"]],
        on=["sourceVideo", "matlabFrameIdx"],
        how="inner"
    )

    # --- Drop unused columns ---
    df_pred = df_pred.drop(["coords", "Unnamed: 61"], axis=1)
    df_gt = df_gt.drop(["coords"], axis=1, errors="ignore")

    # --- Truncate columns if needed ---
    df_pred = df_pred.iloc[:, :60]  # 20 keypoints * 3
    df_gt = df_gt.iloc[:, :40]      # 20 keypoints * 2

    # --- Convert to NumPy arrays ---
    df_pred = df_pred.apply(pd.to_numeric, errors='coerce')
    df_gt = df_gt.apply(pd.to_numeric, errors='coerce')

    n_keypoints = 20
    n_frames = len(df_pred)

    pred = df_pred.to_numpy(dtype=np.float32).reshape(n_frames, n_keypoints, 3)
    gt = df_gt.to_numpy(dtype=np.float32).reshape(n_frames, n_keypoints, 2)

    return pred, gt, mapped_indices
# DONE
def add_visibility(gt:np.ndarray)->np.ndarray:
    # gt has shape: (frames, keypoints, 2 or 3 or 4)
    # Getting the x values for each keypoint for each frame
    x_nan = np.isnan(gt[:, :, 0]) # True if a x value is nan false if not, shape = (frames, keypoints)
    # Getting the y values for each keypoint for each frame
    y_nan = np.isnan(gt[:, :, 1]) # True if a y value is nan false if not, shape = (frames, keypoints)
    # Defining if a point is visible or not 
    visibility = (~np.logical_or(x_nan, y_nan)).astype(np.float32) # if either a keypoint has nan in x or y its not visible 
    # and its set to 0 and if none of them are none its visible and its set to 1.
    
    # Adding the visibility array to the gt so its shape becomes (frames, keypoints, +1)
    return np.concatenate([gt, visibility[:, :, np.newaxis]], axis=2)
# DONE 
# NB! SHOULD WE REALLY HAVE THE WORM BE A PART OF THE BOUNDING BOX THE RESULTS MIGHT BE A LITTLE BIT BETTER THAM THEY ACTUALLY ARE BECAUSE OF THIS!
def compute_area(points:np.ndarray)->float:
    if points.size == 0:
        return 0
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0]) 
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    return (max_x - min_x) * (max_y - min_y)
# DONE
def compute_oks(gt_with_vis: np.ndarray, pred: np.ndarray, sigma: float) -> np.array:
    frames, keypoints, _ = gt_with_vis.shape
    oks_per_frame = np.zeros(frames)

    for f in range(frames):
        gt_visible_flag = gt_with_vis[f, :, 2] == 1
        pred_valid_flag = ~(np.isnan(pred[f, :, 0]) | np.isnan(pred[f, :, 1]))
        valid_mask = gt_visible_flag & pred_valid_flag

        if not np.any(valid_mask):
            oks_per_frame[f] = np.nan
            continue

        ks_scores = np.zeros(keypoints, dtype=np.float32)
        gt_frame = gt_with_vis[f]
        pred_frame = pred[f]
        valid_indices = np.where(valid_mask)[0]

        for k in valid_indices:
            if k == keypoints - 1:
                # Worm: use all visible keypoints
                area_kps = gt_frame[gt_visible_flag, :2]
            else:
                # All other keypoints: exclude the worm
                visible_mask = gt_visible_flag.copy()
                visible_mask[keypoints - 1] = False
                area_kps = gt_frame[visible_mask, :2]

            area = compute_area(area_kps)
            if area == 0:
                continue

            s = np.sqrt(area)
            d = np.linalg.norm(pred_frame[k, :2] - gt_frame[k, :2])
            ks = np.exp(-d**2 / (2 * s**2 * sigma**2))
            ks_scores[k] = ks

        oks_per_frame[f] = np.mean(ks_scores[valid_indices])

    return oks_per_frame
# DONE
def compute_ks(gt_with_vis: np.ndarray, pred: np.ndarray, sigma: float):
    frames, keypoints, _ = gt_with_vis.shape
    ks_per_keypoint = np.zeros((frames, keypoints))

    for f in range(frames):
        gt_visible_flag = gt_with_vis[f, :, 2] == 1 
        pred_valid_flag = ~(np.isnan(pred[f, :, 0]) | np.isnan(pred[f, :, 1]))

        if not np.any(gt_visible_flag):
            continue

        gt_frame = gt_with_vis[f]
        pred_frame = pred[f]

        for k in range(keypoints):
            if not gt_visible_flag[k] or not pred_valid_flag[k]:
                continue

            if k == keypoints - 1:
                # Worm: include all visible keypoints
                area_kps = gt_frame[gt_visible_flag, :2]
            else:
                # Other keypoints: exclude the worm (last keypoint)
                visible_mask = gt_visible_flag.copy()
                visible_mask[keypoints - 1] = False
                area_kps = gt_frame[visible_mask, :2]

            area = compute_area(area_kps)
            if area == 0:
                continue

            d = np.linalg.norm(pred_frame[k, :2] - gt_frame[k, :2])
            s = np.sqrt(area)
            ks = np.exp(-d**2 / (2 * s**2 * sigma**2))
            ks_per_keypoint[f, k] = ks

    return ks_per_keypoint
# DONE
def compute_ap(oks_scores:np.array, threshold:float):
    valid = ~np.isnan(oks_scores) 
    oks_valid = oks_scores[valid] # Only using the oks scores that are not nan
    tp = np.sum(oks_valid >= threshold) # Number of frames with a pose (oks) above the threshold 
    fp = np.sum(oks_valid < threshold) # Number of frames with a pose (oks) below the threshold 
    precision = tp / (tp + fp + np.spacing(1)) if (tp + fp) > 0 else 0
    
    return precision
# DONE
def compute_map(oks_scores:np.array)->np.array:
    """Compute mean AP over multiple thresholds."""
    thresholds = np.arange(0.5, 1.0, 0.05) # Generating a array of threshold values
    aps = [compute_ap(oks_scores, th) for th in thresholds] # computing the ap for each threshold
    
    return np.mean(aps)
# DONE
def evaluate_pose(gt:np.ndarray, pred:np.ndarray, sigma:float)->dict:
    gt_with_vis = add_visibility(gt)
    oks_per_frame = compute_oks(gt_with_vis, pred, sigma)
    mAP = compute_map(oks_per_frame)
    mOKS = np.nanmean(oks_per_frame)
    ks_per_keypoint = compute_ks(gt_with_vis, pred, sigma)
    ap50 = compute_ap(oks_per_frame, 0.50)
    ap75 = compute_ap(oks_per_frame, 0.75)
    ap95 = compute_ap(oks_per_frame, 0.95)

    return {
        'oks_per_frame': oks_per_frame,
        'mAP': mAP,
        'mOKS': mOKS,
        'AP_50': ap50,
        'AP_75': ap75,
        'AP_95': ap95,
        'ks_per_keypoint': ks_per_keypoint
    }
# DONE
def compute_raw_l2_dist(pred_points:np.ndarray, gt_points:np.ndarray)->np.ndarray:
    # Getting the x and y values for predictions for each frame and all keypoints in each frame
    pred_positions = pred_points[:, :, :2]
    visibility_mask = ~np.isnan(gt_points).any(axis=-1) # True if both x and y for a keypoint is nan else false
    # Making a np.ndarray to store the calcualted distances and has shape (frames, keypoints)
    dists = np.full(visibility_mask.shape, np.nan, dtype=np.float32) 
    diffs = pred_positions - gt_points # Calculationg the differnece between gt and preds
    l2_dists = np.linalg.norm(diffs, axis=-1) # Calulationg the euclidain distance 
    dists[visibility_mask] = l2_dists[visibility_mask] # Only assing distances to valid points
    
    # returns a ndarray with shape frames, keypoints where some keypoints have nan values for distances because either no gt or preds right
    return dists
# DONE
def compute_overall_dist_metrics(pred_points:np.ndarray, gt_points:np.ndarray)->dict:
    dists = compute_raw_l2_dist(pred_points, gt_points)
    dists_flat = dists[~np.isnan(dists)]
    
    return {
        "dist.p50": np.percentile(dists_flat, 50),
        "dist.p75": np.percentile(dists_flat, 75),
        "dist.p95": np.percentile(dists_flat, 95)
    }
# DONE
def mean_conf_ks_per_kp_single(pred: np.ndarray, gt: np.ndarray, sigma: float):
    gt_vis = add_visibility(gt[..., :2])
    ks_mat = compute_ks(gt_vis, pred, sigma)
    conf_mat = pred[..., 2].copy()

    # Only keep entries where KS > 0 (i.e., valid match) and GT is visible
    valid_mask = (gt_vis[..., 2] == 1) & (ks_mat > 0)

    # Apply mask to both
    ks_mat[~valid_mask] = np.nan
    conf_mat[~valid_mask] = np.nan

    return np.nanmean(conf_mat, axis=0), np.nanmean(ks_mat, axis=0)

def extract_occlusion_mask(num_frames: int, unocc: int = 25, occ: int = 8) -> np.ndarray:
    pattern = [False] * unocc + [True] * occ
    repeated = (pattern * ((num_frames + len(pattern) - 1) // len(pattern)))[:num_frames]
    return np.array(repeated, dtype=bool)

def evaluate_pose_with_occlusion(gt: np.ndarray, pred: np.ndarray, sigma: float) -> dict:
    gt_with_vis = add_visibility(gt)
    oks_per_frame = compute_oks(gt_with_vis, pred, sigma)

    mOKS_all = np.nanmean(oks_per_frame)
    occlusion_mask = extract_occlusion_mask(len(oks_per_frame))
    mOKS_occ = np.nanmean(oks_per_frame[occlusion_mask])
    mOKS_unocc = np.nanmean(oks_per_frame[~occlusion_mask])

    ks_per_keypoint = compute_ks(gt_with_vis, pred, sigma)
    mAP = compute_map(oks_per_frame)
    ap50 = compute_ap(oks_per_frame, 0.50)
    ap75 = compute_ap(oks_per_frame, 0.75)
    ap95 = compute_ap(oks_per_frame, 0.95)

    return {
        'oks_per_frame': oks_per_frame,
        'mAP': mAP,
        'mOKS': mOKS_all,
        'mOKS_occ': mOKS_occ,
        'mOKS_unocc': mOKS_unocc,
        'AP_50': ap50,
        'AP_75': ap75,
        'AP_95': ap95,
        'ks_per_keypoint': ks_per_keypoint,
        'occlusion_mask': occlusion_mask
    }

# __________________________________________Inference evalutaion:__________________________________________   

# DONE     
def extract_inf_pred_slp(h5_analysis_filepath: str) -> np.ndarray:
    with h5py.File(h5_analysis_filepath, "r") as f:
        preds = f["tracks"][:]          
        conf_scores = f["point_scores"][:] 

    preds = preds[0]                    
    conf_scores = conf_scores[0]        

    preds = np.transpose(preds, (2, 1, 0))  
    conf_scores = conf_scores.T          
    conf_scores_exp = np.expand_dims(conf_scores, axis=-1)

    data = np.concatenate((preds, conf_scores_exp), axis=-1) 

    frame_indices = np.arange(data.shape[0])[:, np.newaxis, np.newaxis]
    frame_indices_repeated = np.repeat(frame_indices, data.shape[1], axis=1)

    result = np.concatenate((data, frame_indices_repeated), axis=-1) 

    return result
# DONE
def extract_inf_pred_lp_dlc(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, header=2)
    frame_indices = df.iloc[:, 0].to_numpy().astype(np.int32)
    keypoint_data = df.iloc[:, 1:].to_numpy().astype(np.float32)

    num_frames = keypoint_data.shape[0]
    num_keypoints = keypoint_data.shape[1] // 3

    keypoint_data = keypoint_data.reshape((num_frames, num_keypoints, 3))

    frame_idx_expanded = frame_indices[:, np.newaxis, np.newaxis]
    frame_idx_repeated = np.repeat(frame_idx_expanded, num_keypoints, axis=1)

    result = np.concatenate((keypoint_data, frame_idx_repeated), axis=-1)
    return result
# DONE
def extract_ap_at(oks_scores, threshold):
    valid = ~np.isnan(oks_scores)
    oks_valid = oks_scores[valid]
    if len(oks_valid) == 0:
        return np.nan
    return np.mean(oks_valid >= threshold)
# DONE
def evaluate_all_videos(gt_data:dict[str, np.ndarray], pred_paths_dict:dict[str, np.ndarray], sigma:float, drop_worm:bool=False)->dict:
    moks  = []
    mAP   = []
    ap50  = []
    ap75  = []
    ap95  = []
    
    common_error_counter = 0

    # For each video and its corresponding set of predictios for a model:
    for vid_name, pred in pred_paths_dict.items():
        if vid_name not in gt_data:
            continue
        
        # Finding common frame indices, grount-truth frame indicies are proxy for when the animal is in frame
        gf = gt_data[vid_name][:, 0, 3].astype(int)
        pf = pred[:, 0, 3].astype(int)
        common = np.intersect1d(gf, pf)
        
        if common.size == 0:
            common_error_counter += 1
            continue
        
        gmask, pmask = np.isin(gf, common), np.isin(pf, common) # True if ground-truth frames are in common and True if predicted frames are in common else False.
        sgt, spr = np.argsort(gt_data[vid_name][gmask][:, 0, 3]), np.argsort(pred[pmask][:, 0, 3]) # Sorting gt and pred data to allign the frames correctly
        
        if drop_worm:
            aligned_gt  = gt_data[vid_name][gmask][sgt][:, :-1, :2]
            aligned_pr  = pred[pmask][spr][:, :-1, :3] 
        else:
            aligned_gt  = gt_data[vid_name][gmask][sgt][:, :, :2]
            aligned_pr  = pred[pmask][spr][:, :, :3]
            
        res = evaluate_pose(aligned_gt, aligned_pr, sigma)

        moks.append(res["mOKS"])
        mAP .append(res["mAP"])
        ap50.append(res["AP_50"])
        ap75.append(res["AP_75"])
        ap95.append(res["AP_95"])

    moks_arr = np.array(moks, dtype=float)
    map_arr  = np.array(mAP , dtype=float)
    
    print(common_error_counter)

    return {
        "mOKS": np.nanmean(moks_arr),
        "mAP": np.nanmean(map_arr),
        "AP_50": np.nanmean(ap50),
        "AP_75": np.nanmean(ap75),
        "AP_95": np.nanmean(ap95),
    }
# DONE
def gather_ks_across_videos(
    gt_data:dict[str, np.ndarray], # ndarray.shape = (frames, keypoints, 4 (x, y, confidence scores, frame idx))
    preds_paths_dict:dict[str, np.ndarray], # ndarray.shape = (frames, keypoints, 4 (x, y, confidence scores, frame idx))
    sigma:float=0.025
)->np.ndarray:    
    ks_chunks = [] # A list to store the ks scores for each video

    for video_name, preds in preds_paths_dict.items(): 
        gt = gt_data[video_name][:, :, :2] # Extracting the x and y corresponding GT-values 
        gt_fidx = gt_data[video_name][:, 0, 3].astype(int) # Extracting the frame idx from the GT-data
        pr_fidx = preds[:, 0, 3].astype(int) # Extracting the frame idx from the pred-data
        common = np.intersect1d(gt_fidx, pr_fidx) # Finding frame idx that are both in GT-data and pred-data

        gt_mask, pr_mask = np.isin(gt_fidx, common), np.isin(pr_fidx, common) # Mask: TRUE if a frame idx is in common FALSE if not
        sgt_idx = np.argsort(gt_data[video_name][gt_mask][:, 0, 3]) # Sorting the common indicies in GT-data
        spr_idx = np.argsort(preds[pr_mask][:, 0, 3]) # Sorting the common indicies in pred-data
        
        # Know the indicies are aligend one to one between GT and pred data:
        aligned_gt = gt_data[video_name][gt_mask][sgt_idx][:, :, :2]
        aligned_pred = preds[pr_mask][spr_idx][:, :, :3]

        # Computing the ks scores
        ks = compute_ks(add_visibility(aligned_gt), aligned_pred, sigma)
        ks_chunks.append(ks)

    return np.vstack(ks_chunks) 
# DONE
def plot_ks_pies_accros_all_videos(model_name:str, ks_matrix:np.ndarray, save_dir:str, good_th:float=0.85, near_miss_th:float=0.5)->None:   
    ks_flat = ks_matrix.flatten()
    valid_mask = ks_flat > 0 # Ignores ks values that are 0               
    ks_valid = ks_flat[valid_mask]

    good_count  = (ks_valid >= good_th).sum() 
    near_miss_count  = ((ks_valid > near_miss_th) & (ks_valid < good_th)).sum()
    miss_count  = (ks_valid <= near_miss_th).sum()
    total = ks_valid.size 

    percentage_function = lambda x: 100 * x / total
    good_percentage, near_miss_percentage, miss_percentage = map(percentage_function, (good_count, near_miss_count, miss_count))

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.pie([good_percentage, near_miss_percentage, miss_percentage],
           labels=["Good", "Near Miss", "Miss"],
           colors=["forestgreen", "orange", "firebrick"],
           explode=[0.02, 0.02, 0.02],
           autopct="%.1f%%",
           startangle=140)
    ax.set_title(f"{model_name}", fontsize=10, y=1.05)
    ax.axis("equal")
    plt.show()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{model_name.replace(' ', '_').replace('/', '_')}_ks_pie.png"
    fig.savefig(Path(save_dir) / fname)

def summarize_top_k_keypoints_across_videos(
    gt_data: dict[str, np.ndarray],
    preds_paths_dict: dict[str, np.ndarray],
    sigma: float,
    top_k: int = 3
):
    keypoint_scores = []

    for vid_name, pred in preds_paths_dict.items():
        if vid_name not in gt_data:
            continue

        gt = gt_data[vid_name][:, :, :2]  # Ground-truth x, y only
        gf = gt_data[vid_name][:, 0, 3].astype(int)  # GT frame indices
        pf = pred[:, 0, 3].astype(int)  # Pred frame indices
        common = np.intersect1d(gf, pf)
        if common.size == 0:
            continue

        gmask, pmask = np.isin(gf, common), np.isin(pf, common)
        sgt, spr = np.argsort(gt_data[vid_name][gmask][:, 0, 3]), np.argsort(pred[pmask][:, 0, 3])
        gt = gt_data[vid_name][gmask][sgt][:, :, :2]
        pr = pred[pmask][spr][:, :, :3]

        F, K = gt.shape[:2]
        ks_mat = np.zeros((F, K), dtype=np.float32)

        vis_gt = ~np.isnan(gt[..., 0]) & ~np.isnan(gt[..., 1])  # Ground-truth visibility mask
        valid_pr = ~np.isnan(pr[..., 0]) & ~np.isnan(pr[..., 1])  # Prediction validity mask

        for f in range(F):
            gt_frame = gt[f]
            pr_frame = pr[f]
            vis_flags = vis_gt[f]
            pred_flags = valid_pr[f]

            for k in range(K):
                if not vis_flags[k] or not pred_flags[k]:
                    continue

                if k == K - 1:
                    # Worm: use full bbox from all visible keypoints
                    area_kps = gt_frame[vis_flags]
                else:
                    # Other keypoints: exclude worm from bbox
                    mask = vis_flags.copy()
                    mask[K - 1] = False
                    area_kps = gt_frame[mask]

                area = compute_area(area_kps)
                if area == 0:
                    continue

                d = np.linalg.norm(pr_frame[k, :2] - gt_frame[k, :2])
                s = np.sqrt(area)
                ks = np.exp(-d**2 / (2 * s**2 * sigma**2))
                ks_mat[f, k] = ks

        keypoint_scores.append(ks_mat)

    if not keypoint_scores:
        return []

    combined = np.concatenate(keypoint_scores, axis=0)  # (ΣF, K)
    mean_ks = np.nanmean(combined, axis=0)
    std_ks = np.nanstd(combined, axis=0)
    worst = np.argsort(mean_ks)[:top_k]

    return [(int(i), float(mean_ks[i]), float(std_ks[i])) for i in worst]

def mean_conf_ks_per_kp(gt_data: dict[str, np.ndarray], preds_dict: dict[str, np.ndarray], sigma: float):
    K = gt_data[next(iter(gt_data))].shape[1]
    conf_acc, ks_acc = [], []

    for vid, pr in preds_dict.items():
        if vid not in gt_data:
            continue

        gf = gt_data[vid][:, 0, 3].astype(int)
        pf = pr[:, 0, 3].astype(int)
        common = np.intersect1d(gf, pf)
        if not common.size:
            continue

        gm, pm = np.isin(gf, common), np.isin(pf, common)
        sgt, spr = np.argsort(gt_data[vid][gm][:, 0, 3]), np.argsort(pr[pm][:, 0, 3])
        gt = gt_data[vid][gm][sgt][:, :, :2]
        pr = pr[pm][spr][:, :, :3]

        gt_vis = add_visibility(gt)
        ks_mat = compute_ks(gt_vis, pr, sigma)
        conf_mat = pr[..., 2].copy()

        # Only keep entries where GT is visible and KS > 0
        valid_mask = (gt_vis[..., 2] == 1) & (ks_mat > 0)

        ks_mat[~valid_mask] = np.nan
        conf_mat[~valid_mask] = np.nan

        ks_acc.append(ks_mat)
        conf_acc.append(conf_mat)

    if not ks_acc:
        nan = np.full(K, np.nan)
        return nan, nan

    return (
        np.nanmean(np.concatenate(conf_acc, axis=0), axis=0),
        np.nanmean(np.concatenate(ks_acc, axis=0), axis=0),
    )


