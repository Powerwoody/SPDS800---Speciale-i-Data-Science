import os
import argparse
import subprocess

def main(input_dir, model_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".mp4"):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.slp")

            # Build sleap-track command
            cmd = ["sleap-track"]
            for model in model_paths:
                cmd.extend(["-m", model])
            cmd.extend(["-o", output_path, input_path])

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch track videos using multiple SLEAP models.")
    parser.add_argument("--input_dir", required=True, help="Directory with input .mp4 videos.")
    parser.add_argument("--model_path", nargs='+', required=True, help="Paths to one or more SLEAP models.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output .slp files.")

    args = parser.parse_args()
    main(args.input_dir, args.model_path, args.output_dir)
