📊 Project Overview
This project, "Hand Gesture Recognition System," focuses on developing a real-time gesture-based communication system using MediaPipe, OpenCV, and LSTM Neural Networks. The system aims to enhance human-computer interaction by accurately recognizing hand gestures, particularly for speech-impaired individuals. By leveraging AI-driven methodologies, the project facilitates gesture-to-text translation, improving accessibility and real-time communication.

👥 Team Members
Soham Vasudeo
Om Agrawal
Vanshaj Ajmera
Shardul Patki
Ritika Shetty
Course: Machine Learning Project 2022-23
Institution: NMIMS University
🎯 Objectives
The objective of this project is to create a robust real-time hand gesture recognition system that can accurately interpret predefined gestures and expand to accommodate new gestures for various applications.

Key Goals:
✔ Enable gesture-based communication for speech-impaired individuals.
✔ Implement a real-time detection system with high accuracy and low latency.
✔ Develop a scalable and adaptable model to recognize additional gestures.
✔ Enhance human-computer interaction through gesture-controlled systems.

🗂 Data Understanding
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
📌 Data Storage: Organized in labeled directories for training, validation, and testing.
Challenges Addressed:
🚧 Variations in Lighting & Backgrounds – Improved model robustness to different environments.
🚧 Real-Time Processing Constraints – Optimized inference time for faster recognition.
🚧 Multi-Gesture Recognition – Designed the model to learn new gestures dynamically.

⚙️ Data Preparation
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

🧠 Modeling
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

📈 Model Evaluation
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

📊 Key Insights and Results
📌 High Recognition Accuracy: 96.4% accuracy on test data.
📌 Fast Processing Speed: Real-time recognition at ~3.81 ms per frame.
📌 User-Friendly Interface: Seamless gesture-to-text translation for assistive communication.
📌 Expandable Gesture Set: New gestures can be added without retraining the entire model.

🛠 Tools & Technologies Used
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


🏆 Achievements
🏅 Developed a fully functional real-time gesture recognition system.
🏅 Achieved 96.4% accuracy using deep learning models.
🏅 Successfully integrated MediaPipe for precise hand tracking.
🏅 Enabled real-time gesture classification for assistive communication.


📊 Project Overview
This project, "Transforming Financial Planning and Analysis (FP&A) with Generative AI," explores the integration of Robotic Process Automation (RPA), deep learning, and Generative AI to enhance the efficiency and accuracy of financial forecasting and analysis. By modernizing traditional FP&A processes, we aim to enable organizations to expedite decision-making, optimize costs, and respond dynamically to market fluctuations.

👥 Team Members
Kavit Navadia
Shardul Patki
Soham Vasudeo
Course: BIT 5524 - Conceptual Project Report
Institution: Virginia Tech

🎯 Objectives
The objective of this project is to modernize financial planning and analysis by automating data collection, improving forecasting accuracy, and providing real-time actionable insights using AI-driven methodologies.

Key Goals:

Automate FP&A processes to reduce manual effort.
Improve forecasting accuracy using machine learning models.
Optimize resource allocation through data-driven decision-making.
Proactively detect financial anomalies to mitigate risk.
🗂 Data Understanding
The project leverages various data sources to improve financial forecasting accuracy, including:

Historical Financial Data:

Profit and Loss statements
Balance Sheets
Cash Flow statements
Budget vs Actuals Analysis
Market and External Data:

Inflation rates, GDP, and interest rates
Competitor performance benchmarks
AI-Generated Data:

Simulated market conditions
Scenario-based financial projections
Challenges Addressed:

Data completeness, consistency, and accuracy issues
Standardization across different formats and currencies
Integration of real-time updates from ERP systems
⚙️ Data Preparation
The data preparation phase includes:

Data Cleaning:

Handling missing values using imputation techniques.
Standardizing currency and date formats.
Removing redundant records.
Data Transformation:

Scaling and encoding data for modeling.
Feature engineering to derive new insights such as growth rates and revenue ratios.
Data Splitting:

Training (70%), Validation (15%), and Testing (15%) to avoid data leakage.
🧠 Modeling
We utilized advanced machine learning models to provide predictive insights:

1. Time-Series Forecasting Models:
LSTM (Long Short-Term Memory): Captures sequential dependencies for long-term predictions.
GRU (Gated Recurrent Units): Efficient for real-time forecasting updates.
SARIMA (Seasonal AutoRegressive Integrated Moving Average): Incorporates seasonality patterns for financial data.
2. Anomaly Detection:
Isolation Forest: Identifies outliers in financial metrics.
Autoencoders: Detect unexpected fluctuations in financial data.
3. Generative AI for Scenario Analysis:
Used LLMs (Large Language Models) such as OpenAI API to analyze qualitative financial insights.
Integrated Reinforcement Learning with Human Feedback (RLHF) to improve forecasting accuracy.
📈 Model Evaluation
The models were evaluated based on the following metrics:

Time-Series Forecasting Metrics:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Anomaly Detection Metrics:

Precision, Recall, and F1-score for anomaly identification.
Generative AI Evaluation:

Relevance and accuracy of AI-generated insights.
User satisfaction scores and governance compliance.
🚀 Deployment Strategy
The deployment phase integrates the AI-driven FP&A system into enterprise workflows with a focus on scalability and security. The deployment components include:

Infrastructure Setup:

AWS (S3, Redshift), Azure (Blob Storage), or Google Cloud for hosting and storage.
Robotic Process Automation (RPA):

Tools like UiPath and Power Automate to collect and clean financial data.
Data Pipelines (ETL):

Automated ETL processes using Apache Airflow and dbt for data transformation.
Model Deployment:

Deploying models via AWS SageMaker or Azure ML for real-time forecasting.
Monitoring & Maintenance:

Continuous evaluation using CloudWatch and Azure Monitor for real-time insights and retraining.
📊 Key Insights and Results
Reduction in manual effort: By automating data collection and reporting.
Improved accuracy: Forecasting error reduced by 15% with AI-driven predictions.
Cost Optimization: Identified cost-saving opportunities using anomaly detection.
Scenario Planning: "What-if" analysis improved strategic decision-making.
🛠 Tools & Technologies Used
Programming & Frameworks: Python, Pandas, NumPy, TensorFlow, Scikit-learn
Data Storage: AWS S3, Snowflake, Google BigQuery
Visualization: Power BI, Plotly, Tableau
AI/ML Models: LSTM, GRU, SARIMA, Isolation Forest
Automation Tools: UiPath, Power Automate, Apache Airflow
Cloud Services: AWS, Azure, Google Cloud
LLM Integration: OpenAI API, LangChain
📊 Visuals
image

🏆 Achievements
Developed a fully automated financial forecasting pipeline with AI.
Created dynamic dashboards for real-time insights.
Implemented anomaly detection to identify financial risks proactively.
