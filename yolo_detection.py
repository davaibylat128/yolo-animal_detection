import cv2
import torch
import numpy as np
import pygame

# Initialize the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Update the target classes to include common animals
target_classes = ['cow', 'horse', 'deer', 'wild boar', 'monkey', 'elephant', 'brown bear']

# Initialize pygame
pygame.init()
pygame.mixer.init()  # Ensure the mixer module is initialized

# Load the alarm sound
pygame.mixer.music.load("static/alarm.wav")

def detect_animals(frame):
    results = model(frame)

    # Using pandas to get the detected objects' data
    for index, row in results.pandas().xyxy[0].iterrows():
        if row['name'] in target_classes:
            name = str(row['name'])
            x1, y1 = int(row['xmin']), int(row['ymin'])
            x2, y2 = int(row['xmax']), int(row['ymax'])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            # Write name
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # Draw center
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Play the alarm sound if an animal is detected
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

            cv2.putText(frame, "Animal Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return frame

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img
