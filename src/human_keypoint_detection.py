"""
human_keypoint_detection.py

This script performs human keypoint detection on a video using YOLO model.
Provide the path to the YOLO model and the input video in a configuration file.
Currently, uses the default yolo pose model. Will download the default model automatically if not found.
"""
import yaml
from ultralytics import YOLO

def process_video(config):
    # Load a model
    model = YOLO(config['yolo']['model_path'])

    # Predict with the model
    results = model(
        config['prediction']['input_path'],
        show=config['prediction']['show'],
        save=config['prediction']['save'],
        show_boxes=config['prediction']['show_boxes'],
        show_labels=config['prediction']['show_labels'],
        show_conf=config['prediction']['show_conf'],
        conf=config['prediction']['conf']
    )

def main():
    """
    Main function to process the video.
    """
    try:
        with open("../config/human_keypoint_detection_config,yaml", 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Configuration file not found.")
        return

    process_video(config)

if __name__ == "__main__":
    main()
