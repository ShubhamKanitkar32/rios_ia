"""
region_counting.py

This script uses the YOLO model for object counting in a region in a video file.
The video is processed frame by frame, and the counted objects are shown in the video.
The user has an ability to move the counting regions in the video.
The paths for the model and the input and output videos, as well as the region counting configurations, are specified in a configuration file.
"""
import os
import logging
import yaml
import cv2
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np

track_history = defaultdict(list)
current_region = None

def mouse_callback(event, x, y, flags, param):
    global current_region

    if event == cv2.EVENT_LBUTTONDOWN:
        for region in param:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False

paused = False

def process_video(config):
    global paused

    weights = config['yolo']['model_path']
    source = config['video']['input_path']
    device = config['yolo']['device']
    view_img = config['counter']['view_img']
    save_img = config['video']['save_output']
    line_thickness = config['counter']['line_thickness']
    region_thickness = config['counter']['region_thickness']

    counting_regions = [
        {
            "name": region['name'],
            "polygon": Polygon(region['points']),
            "counts": 0,
            "dragging": False,
            "region_color": tuple(region['region_color']),
            "text_color": tuple(region['text_color']),
            "display_text": region['display_text'],
        } for region in config['regions']
    ]

    # Model Setup
    model = YOLO(weights)
    model.to("cuda") if device == "0" else model.to("cpu")
    names = model.model.names

    # Video Setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output Setup
    save_dir = Path(config['video']['output_path'])
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    while videocapture.isOpened():
        if not paused:
            success, frame = videocapture.read()
            if not success:
                break

            results = model.track(frame, persist=True, classes=None, show=False, show_boxes=False, show_labels=False, show_conf=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                masks = results[0].masks.xy
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()

                annotator = Annotator(frame, line_width=line_thickness, example=str(names))

                for box, mask, track_id, cls in zip(boxes, masks, track_ids, clss):
                    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                    for region in counting_regions:
                        if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                            region["counts"] += 1

            for region in counting_regions:
                region_label = f"{region['display_text']}: {region['counts']}"
                region_color = region["region_color"]
                region_text_color = region["text_color"]

                transparent_frame = frame.copy()  # Create a copy of the frame for drawing transparent polygon
                transparent_frame = cv2.fillPoly(transparent_frame, [np.array(region["polygon"].exterior.coords, dtype=np.int32)], color=(0, 0, 255, 100))  # Draw semi-transparent polygon

                # Blend the transparent frame with the original frame
                frame = cv2.addWeighted(transparent_frame, 0.5, frame, 0.7, 0)

                centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

                text_size, _ = cv2.getTextSize(
                    region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
                )
                text_x = centroid_x - text_size[0] // 2
                text_y = centroid_y + text_size[1] // 2
                cv2.putText(
                    frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
                )

            if view_img:
                if not paused:
                    if videocapture.get(cv2.CAP_PROP_POS_FRAMES) == videocapture.get(cv2.CAP_PROP_FRAME_COUNT):
                        videocapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
                    cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback, counting_regions)
                    cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

            if save_img:
                video_writer.write(frame)

            for region in counting_regions:
                region["counts"] = 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused

    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to process the video.
    """
    try:
        with open('../config/region_counting_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(config['video']['output_path'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_video(config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
