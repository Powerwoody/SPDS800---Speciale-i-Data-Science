import deeplabcut as dlc
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze a video using a DLC project.")
    parser.add_argument("project_path", type=str, help="Path to the DLC project directory")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze")
    args = parser.parse_args()

    project_path = args.project_path.strip()
    video_path = args.video_path.strip()

    # Tjek om mappen og filen eksisterer
    if not os.path.isdir(project_path):
        raise FileNotFoundError(f"The provided project path does not exist: {project_path}")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"The provided video file does not exist: {video_path}")

    config_path = os.path.join(project_path, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.yaml found at: {config_path}")

    try:
        dlc.analyze_videos(
            config=config_path,
            videos=[video_path],
            videotype=".mp4",
            save_as_csv=True
        )
        print("Video analysis completed successfully.")
    except Exception as e:
        print(f"Error during video analysis: {e}")

if __name__ == "__main__":
    main()
