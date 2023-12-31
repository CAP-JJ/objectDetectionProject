# Real-Time Object Detection with YOLO and RealSense Depth Camera

## Overview

This project implements real-time object detection and segmentation using the YOLO (You Only Look Once) model, integrated with a RealSense camera. The script captures video frames from the RealSense camera, applies object detection, overlays segmentation masks, and visualizes the results in real-time.

## Features

- Real-time object detection using YOLO model.
- Overlaying of segmentation masks on detected objects.
- Real-time visualization of detection results.

## Requirements

To run this project, you will need the following:

- Python 3.11
- OpenCV 4.8 or later
- pyrealsense2 2.54.2 or later
- NumPy 1.26 or later
- Ultralytics 8.0 or later
- Intel RealSense camera

## Installation

1. Clone this repository to your local machine using `git clone https://github.com/CAP-JJ/objectDetectionProject.git`
2. Set the default directory to the local repo using `cd <repo>`
3. Install virtualenv using `pip install virtualenv`
4. Create a new virtual environment using `virtualenv venv`
5. Activate the new virtual environment using `source venv/bin/activate`
6. Install the requirements in the new environment using `pip install -r requirements.txt`

## Usage

1. Run the `main.py` script to start the object detection process.
2. The script will use the Intel RealSense camera to capture depth data and images.
3. YOLO models will be used to detect objects in the images.
4. The detected objects will be displayed on the screen.

## Documentation

Alongside the provided information throughout the code, a Jupyter Notebook is added as a presentation. Additionally, this presentation can be accessed on GitHub pages.

## Code Structure

- overlay: Function to overlay segmentation masks on images.
- plot_one_box: Function to plot bounding boxes around detected objects.
- RealSense camera initialization and configuration.
- Main loop for capturing frames and processing them through the YOLO model.

## Customization
You can customize the detection parameters and model within the script to suit different use cases or improve performance.

## Acknowledgements

Ultralytics for the YOLO model.
Intel RealSense for camera technology.
