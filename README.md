## 📊 Project Overview
This project, "Hand Gesture Recognition System," focuses on developing a real-time gesture-based communication system using MediaPipe, OpenCV, and LSTM Neural Networks. The system aims to enhance human-computer interaction by accurately recognizing hand gestures, particularly for speech-impaired individuals. By leveraging AI-driven methodologies, the project facilitates gesture-to-text translation, improving accessibility and real-time communication.

## 👥 Team Members
Soham Vasudeo
Om Agrawal
Vanshaj Ajmera
Shardul Patki
Ritika Shetty
Course: Machine Learning Project 2022-23
Institution: NMIMS University

## 🎯 Objectives
The objective of this project is to create a robust real-time hand gesture recognition system that can accurately interpret predefined gestures and expand to accommodate new gestures for various applications.

Key Goals:
✔ Enable gesture-based communication for speech-impaired individuals.
✔ Implement a real-time detection system with high accuracy and low latency.
✔ Develop a scalable and adaptable model to recognize additional gestures.
✔ Enhance human-computer interaction through gesture-controlled systems.

## 🗂 Data Understanding
The project leverages gesture-based image data to train and optimize the model for accurate recognition.

Dataset Components:
📌 Hand Gesture Images: Captured using MediaPipe Hands for training the model.
📌 Keypoints Extraction: 21 hand landmarks recorded for precise motion tracking.
📌 Predefined Gestures: The system recognizes five primary gestures:

Hello 🖐
I Love You 🤟
Thank You ✋
One ☝
Victory ✌

## 📌 Data Storage: Organized in labeled directories for training, validation, and testing.
Challenges Addressed:
🚧 Variations in Lighting & Backgrounds – Improved model robustness to different environments.
🚧 Real-Time Processing Constraints – Optimized inference time for faster recognition.
🚧 Multi-Gesture Recognition – Designed the model to learn new gestures dynamically.

## ⚙️ Data Preparation
Data Cleaning & Processing:
🔹 Removed noisy and distorted images.
🔹 Standardized gesture images to a consistent resolution.
🔹 Applied image augmentation (rotation, scaling) to improve model generalization.

Feature Engineering:
🔹 Extracted hand keypoints (x, y, z coordinates) from video frames.
🔹 Converted gesture data into structured NumPy arrays for training.

Data Splitting:
📌 Training Set – 70%
📌 Validation Set – 15%
📌 Testing Set – 15%

## 🧠 Modeling
The project utilizes deep learning models to classify hand gestures based on extracted keypoints.

1. Hand Tracking & Feature Extraction:
✅ MediaPipe Hands: Detects 21 keypoints for each hand.
✅ OpenCV Preprocessing: Converts images into feature arrays for analysis.

2. Deep Learning Models for Gesture Classification:
🤖 LSTM (Long Short-Term Memory): Captures temporal dependencies in hand movements.
🤖 CNN (Convolutional Neural Networks): Extracts spatial patterns for accurate recognition.
🤖 ANN (Artificial Neural Networks): Classifies gesture patterns with optimized weight distribution.

3. Model Training & Fine-Tuning:
✔ Hyperparameter Optimization: Adjusted learning rate, batch size, and epochs for best results.
✔ Loss Function & Optimization: Categorical Cross-Entropy with Adam Optimizer.

## 📈 Model Evaluation
The performance of the gesture recognition system was evaluated using multiple metrics:

Classification Metrics:
📌 Accuracy: 87.5% on the test dataset.
📌 Precision & Recall: High precision ensures minimal false positives.
📌 F1-Score: Balanced assessment of precision and recall.

Inference Speed:
⏳ Processing Time: ~3.81 ms per frame (real-time execution).

🚀 Deployment Strategy
The gesture recognition system is integrated into a real-time application for gesture-to-text translation and smart control systems.

1. Infrastructure Setup:
☁ Compatible with Edge & Cloud Systems – Supports Raspberry Pi, AWS, Google Cloud.

2. Data Processing & Storage:
🔄 Automated Data Collection & Preprocessing for scalability.

3. Model Deployment:
✅ Flask API / FastAPI – Enables real-time communication.
✅ TFLite for Mobile Deployment – Optimized for mobile & embedded devices.

4. Continuous Monitoring & Updates:
📊 Feedback Loop: Allows users to add new gestures dynamically.

## 📊 Key Insights and Results
📌 High Recognition Accuracy: 96.4% accuracy on test data.
📌 Fast Processing Speed: Real-time recognition at ~3.81 ms per frame.
📌 User-Friendly Interface: Seamless gesture-to-text translation for assistive communication.
📌 Expandable Gesture Set: New gestures can be added without retraining the entire model.

## 🛠 Tools & Technologies Used
Programming & Frameworks:
🖥 Python, NumPy, OpenCV, TensorFlow, Keras

Machine Learning Models:
🧠 LSTM, CNN, ANN

Hand Tracking & Data Processing:
🖐 MediaPipe Hands, OpenCV, NumPy

Visualization & Analytics:
📊 Matplotlib, Seaborn, Power BI

Deployment & Integration:
🌍 Flask, FastAPI, TensorFlow Lite (TFLite)

📊 Visuals
📸 Dataset Samples: Hand gesture images captured for training.
![image](https://github.com/user-attachments/assets/a5ad327e-c702-466f-801f-9e95786f7465)
![image](https://github.com/user-attachments/assets/6fd2a07c-5338-4eca-b67e-815261f0477f)
![image](https://github.com/user-attachments/assets/a3594837-85c8-474b-a89f-4c76b695da6b)
![image](https://github.com/user-attachments/assets/597165f9-e38c-4170-89e6-7ede9ad19d47)
![image](https://github.com/user-attachments/assets/ab9470a6-c44e-4022-abd9-141ddaa04a83)

📊 Model Performance Graphs: Accuracy, loss curves, and real-time testing results.
![image](https://github.com/user-attachments/assets/18206e2f-37a6-45d1-a14e-85785dcb8883)


## 🏆 Achievements
🏅 Developed a fully functional real-time gesture recognition system.
🏅 Achieved 96.4% accuracy using deep learning models.
🏅 Successfully integrated MediaPipe for precise hand tracking.
🏅 Enabled real-time gesture classification for assistive communication.
