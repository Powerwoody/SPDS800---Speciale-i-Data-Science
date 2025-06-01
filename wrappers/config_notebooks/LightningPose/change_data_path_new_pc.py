from omegaconf import OmegaConf
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run LP training from a given project path.")
    parser.add_argument("project_path", type=str, help="Path to the LP project directory")
    args = parser.parse_args()

    project_path = args.project_path.strip()
    video_path = os.path.join(project_path, "videos")

    yaml_files = [f for f in os.listdir(project_path) if f.endswith('.yaml')]
    config_path = os.path.join(project_path, yaml_files[0])
    print(config_path)

    cfg = OmegaConf.load(config_path)

    cfg.data.data_dir = project_path
    cfg.data.video_dir = video_path
    
    OmegaConf.save(cfg, config_path)
    
if __name__ == "__main__":
    main()
