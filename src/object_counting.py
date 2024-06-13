"""
object_counting.py

This script uses the YOLO model for object counting in a video file.
The video is processed frame by frame, and the counted objects are saved to a new video file.
The paths for the model and the input and output videos, as well as the counting configurations, are specified in a configuration file.
"""
import os
import logging
import yaml
import cv2
from ultralytics import YOLO, solutions

def process_video(model_path, input_path, output_path, region_points, counter_config):
    """
    Process a video using object counting.

    Args:
        model_path (str): Path to the YOLO model.
        input_path (str): Path to the input video.
        output_path (str): Path to the output video.
        region_points (list): List of region points for object counting.
        counter_config (dict): Dictionary of counter configurations.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counter = solutions.ObjectCounter(
        view_img=counter_config['view_img'],
        reg_pts=region_points,
        classes_names=model.names,
        line_thickness=counter_config['line_thickness'],
        region_thickness=counter_config['region_thickness'],
        view_in_counts=counter_config['view_in_counts'],
        view_out_counts=counter_config['view_out_counts'],
        count_reg_color=tuple(counter_config['count_reg_color'])
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            logging.info("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False, show_boxes=False, show_labels=False, show_conf=False)

        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)

        # Display the resulting frame
        cv2.imshow('Object Counting', im0)

        # If 'q' is pressed on the keyboard,
        # exit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


def main():
    """
    Main function to process the video.
    """
    try:
        with open('../config/object_counting_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        return

    model_path = config['yolo']['model_path']
    input_path = config['video']['input_path']
    output_path = config['video']['output_path']
    region_points = [tuple(point) for point in config['region']['points']]
    counter_config = config['counter']

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_video(model_path, input_path, output_path, region_points, counter_config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()