import deeplabcut as dlc
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run DLC training from a given project path.")
    parser.add_argument("project_path", type=str, help="Path to the DLC project directory")
    args = parser.parse_args()

    project_path = args.project_path.strip()

    if not os.path.isdir(project_path):
        raise FileNotFoundError(f"The provided project path does not exist: {project_path}")

    config_path = os.path.join(project_path, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.yaml found at: {config_path}")

    try:
        dlc.auxiliaryfunctions.edit_config(config_path, {"project_path": project_path})
    except Exception as e:
        print(f"Error editing config file: {e}")
        return

    try:
        dlc.train_network(config_path)
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    dlc.evaluate_network(config_path, plotting=True)

    print("Training is done!")

if __name__ == "__main__":
    main()

        
