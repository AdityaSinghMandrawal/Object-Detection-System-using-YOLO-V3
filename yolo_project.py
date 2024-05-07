import cv2
import numpy as np
from gtts import gTTS
import pygame
import os
import tempfile

# Initialize Pygame mixer
pygame.mixer.init()

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    file_path = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
    tts.save(file_path)

    # Load the audio file and play it using Pygame mixer
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    # Wait for the playback to finish
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        clock.tick(10)  # Control playback speed

    # Stop and unload the audio file from the mixer
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

yolov3_cfg_path = r"C:\Users\Acer\Desktop\Object_detection\yolov3 (1).cfg"
coco_names_path = r"C:\Users\Acer\Desktop\Object_detection\coco.names"
yolov3_weights_path = r"C:\Users\Acer\Desktop\Object_detection\yolov3.weights"

def object_detection():
    # Load the pre-trained YOLOv3 model
    net = cv2.dnn.readNet(yolov3_weights_path, yolov3_cfg_path)
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayersNames()

    # Load COCO labels
    with open(coco_names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Start capturing video from webcam (0 for default webcam)
    cap = cv2.VideoCapture(0)


    detected_objects = set()  # To store detected objects and avoid repetitive speech

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, channels = frame.shape

        # Object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        object_labels = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    object_labels.append(classes[class_id])

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        detected_objects.clear()  # Clear previous detected objects for the new frame

        for i in range(len(boxes)):
            if i in indexes:
                label = str(object_labels[i])
                detected_objects.add(label)  # Add detected object to the set
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y + 30), font, 3, (0, 255, 0), 3)

        if detected_objects:
            text_to_speech(', '.join(detected_objects))

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    object_detection()
