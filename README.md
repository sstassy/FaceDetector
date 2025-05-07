# FaceDetector

Python application inspired by @nicknochnack's Object Detection course.

The project implements a deep learning pipeline for face recognition using data retrieved from a video file as the training source.

A convolutional neural network (CNN) based on the VGG16 architecture is used to detect the presence of faces and predict bounding box coordinates.

---

# Real-Time Face Detection using TensorFlow and OpenCV

This project implements a real-time face detection system using a custom-trained neural network built on top of the VGG16 architecture. The model is trained to classify whether a face is present in a frame and to predict the corresponding bounding box coordinates.

The detection pipeline captures frames from a live webcam feed, resizes and normalizes each frame, and performs inference using a pre-trained Keras model (`facetracker.h5`). When a face is detected with sufficient confidence, the program draws a bounding box and label around the face in real time using OpenCV.

The program (.py) is designed to run in standard desktop Python environments (not in notebooks or WSL) and is optimized for low-latency applications such as vision-based user interfaces or automated monitoring.
