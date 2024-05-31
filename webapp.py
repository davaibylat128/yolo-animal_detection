import argparse
import io
from PIL import Image
import datetime
from waitress import serve
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob

from yolo_detection import detect_animals, preprocess, stop_alarm

from ultralytics import YOLO


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/webcam")
def webcam():
    return render_template('webcam.html')  

@app.route('/web_feed')
def web_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm_route():
    stop_alarm()
    return '', 204

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess(frame)
        frame = detect_animals(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route("/", methods=["GET", "POST"])
def predict_img():
    image_path = None

    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'jpg':
                img = cv2.imread(filepath)

                # Perform the detection
                model = YOLO('best.pt')
                results = model(img, save=True)

                res_plotted = results[0].plot()

                # Save the image with detections
                detected_img_path = os.path.join('static', 'detected.jpg')
                cv2.imwrite(detected_img_path, res_plotted)
                image_path = detected_img_path

            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                model = YOLO('best.pt')
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, save=True)
                    res_plotted = results[0].plot()
                    out.write(res_plotted)
                    if cv2.waitKey(1) == ord('q'):
                        break
                cap.release()
                out.release()

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    image_path = folder_path + '/' + latest_subfolder + '/' + f.filename
    return render_template('index.html', image_path=image_path)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


def get_frame():
    video_path = 'static/output.mp4'
    cap = cv2.VideoCapture(video_path)  # detected video path
    while True:
        success, image = cap.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)  # control the frame rate to display one frame every 100 milliseconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO('best.pt')
    app.run(host="0.0.0.0", port=args.port)
    serve(app, host="0.0.0.0", port=args.port)


