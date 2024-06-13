"""
region_monitoring.py

This script performs region monitoring on a video using YOLO object detection.
It detects objects within a specified region of interest and displays the status
of the region (Idle or Engaged) on a status bar overlayed on the video frames.
"""
import cv2
import numpy as np
import yaml
import os
from ultralytics import YOLO

def is_region_engaged(poly, box):
    point = (float(box[0]), float(box[1]))
    return cv2.pointPolygonTest(poly, point, False) == 1

def process_video(config):
    model = YOLO(config['yolo']['model_path'])

    region_to_check = np.array(config['region']['points'])

    input_video_path = config['video']['input_path']
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*config['video']['codec'])
    output_video_path = config['video']['output_path']
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height + 100))

    status_bar = np.zeros((100, frame_width, 3), dtype=np.uint8)
    fill_alpha = config['region']['fill_alpha']

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracks = model.track(frame, persist=True, show=False, show_boxes=False, show_labels=False, show_conf=False)
        region_status = "Idle"
        for box in tracks[0].boxes.xyxy:
            if is_region_engaged(region_to_check, box):
                region_status = "Engaged"
                break

        status_bar[:] = 0
        status_color = tuple(config['status_bar']['engaged_color'] if region_status == "Engaged" else config['status_bar']['idle_color'])
        cv2.putText(status_bar, f"Lug Loader Status: {region_status}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        frame_with_status_bar = np.vstack((status_bar, frame))

        region_filled = frame_with_status_bar.copy()
        fill_color = tuple(config['region']['engaged_color'] if region_status == "Engaged" else config['region']['idle_color'])
        cv2.fillPoly(region_filled, [region_to_check], fill_color)
        cv2.addWeighted(region_filled, fill_alpha, frame_with_status_bar, 1 - fill_alpha, 0, frame_with_status_bar)

        cv2.polylines(frame_with_status_bar, [region_to_check], isClosed=True, color=fill_color, thickness=config['region']['line_thickness'])

        cv2.imshow('Frame', frame_with_status_bar)
        out.write(frame_with_status_bar)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to process the video.
    """
    try:
        with open("../config/region_monitoring_config.yaml", 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Configuration file not found.")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(config['video']['output_path'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_video(config)

if __name__ == "__main__":
    main()
