# Object Detection System for Visually Impaired Individuals

This system utilizes the YOLOv3 algorithm, COCO dataset, OpenCV, and Google Text-to-Speech (GTTS) to detect objects in real-time from a webcam feed and provide auditory feedback to visually impaired users.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Pygame
- gTTS (Google Text-to-Speech)

You can install the required packages using pip:

## Usage

1. Ensure all requirements are installed.
2. Download the YOLOv3 weights (`yolov3.weights`), configuration file (`yolov3.cfg`), and COCO names file (`coco.names`). Modify the paths in the code to point to these files.
3. Run the `object_detection.py` script.
4. The webcam feed will open, and objects detected in the frame will be announced audibly using GTTS.

## Customization

- You can adjust the confidence threshold for object detection by modifying the `confidence > 0.5` condition in the code.
- To customize the speech output or language, you can modify the `text_to_speech` function in the code.

## Acknowledgments

- YOLOv3: https://pjreddie.com/darknet/yolo/
- COCO Dataset: https://cocodataset.org/
- OpenCV: https://opencv.org/
- GTTS: https://gtts.readthedocs.io/en/latest/


