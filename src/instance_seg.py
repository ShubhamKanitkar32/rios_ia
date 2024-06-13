"""
instance_seg.py

This script uses the YOLO model for instance segmentation on a video file.
The video is processed frame by frame, and the segmented output is saved to a new video file.
The paths for the model and the input and output videos are specified in a configuration file.
"""

import os
import logging
import yaml
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Constants
QUIT_KEY = "q"
QUIT_KEY_ASCII = 0xFF

def process_video(model_path, input_path, output_path):
    """
    Process a video using instance segmentation.

    Args:
        model_path (str): Path to the YOLO model.
        input_path (str): Path to the input video.
        output_path (str): Path to the output video.
    """
    model = YOLO(model_path)
    names = model.model.names
    cap = cv2.VideoCapture(input_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    while True:
        ret, im0 = cap.read()
        if not ret:
            logging.info("Video frame is empty or video processing has been successfully completed.")
            break

        results = model.predict(im0)
        annotator = Annotator(im0, line_width=2)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            for mask, cls in zip(masks, clss):
                annotator.seg_bbox(mask=mask, mask_color=colors(int(cls), True), det_label=names[int(cls)])

        out.write(im0)
        cv2.imshow("instance-segmentation", im0)

        if cv2.waitKey(1) & QUIT_KEY_ASCII == ord(QUIT_KEY):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main function to process the video.
    """
    try:
        with open('../config/instance_seg_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        return

    model_path = config['yolo']['model_path']
    input_path = config['video']['input_path']
    output_path = config['video']['output_path']

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    process_video(model_path, input_path, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()