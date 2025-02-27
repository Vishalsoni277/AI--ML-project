import cv2 
import numpy as np
import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import joblib
import csv
from tensorflow.keras.models import load_model

# Paths for required models
svm_model_path = "models/gender_svm_model.pkl"
scaler_path = "models/scaler.pkl"
feature_extractor_path = "models/feature_extractor.h5"
emotion_model_path = "models/abc.h5"  # Path to your emotion detection model

# Check if all models exist
for path in [svm_model_path, scaler_path, feature_extractor_path, emotion_model_path]:
    if not os.path.exists(path):
        print(f"Error: Model file {path} not found.")
        exit()

# Load the models
svm_model = joblib.load(svm_model_path)
scaler = joblib.load(scaler_path)
feature_extractor = load_model(feature_extractor_path)
emotion_model = load_model(emotion_model_path)  # Load emotion detection model

# Emotion labels
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create directories for storing images
male_dir = "image/capture_dataset/male"
female_dir = "image/capture_dataset/female"
os.makedirs(male_dir, exist_ok=True)
os.makedirs(female_dir, exist_ok=True)

# Create or open the log CSV file
log_file_path = "gender_detection_log.csv"
log_exists = os.path.exists(log_file_path)
with open(log_file_path, mode='a', newline='') as log_file:
    csv_writer = csv.writer(log_file)
    if not log_exists:
        csv_writer.writerow(["Image Name", "Gender", "Accuracy", "Emotion"])

# Initialize counters for male and female images
male_count = len(os.listdir(male_dir))
female_count = len(os.listdir(female_dir))

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Scanning for 10 seconds...")
start_time = time.time()
last_face = None

while time.time() - start_time < 10:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        last_face = frame, faces  # Store last detected face

    # Display the frame
    cv2.imshow('Real-Time Gender and Emotion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if last_face:
    frame, faces = last_face
    x, y, w, h = faces[0]  # Process the first detected face
    try:
        face_roi = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (64, 64))
        face_normalized = face_resized.astype("float32") / 255.0
        face_normalized = np.expand_dims(face_normalized, axis=0)

        # Predict features and scale them
        features = feature_extractor.predict(face_normalized)
        features_scaled = scaler.transform(features)

        # Predict gender
        gender_pred = svm_model.predict(features_scaled)
        accuracy = svm_model.decision_function(features_scaled)[0] + 90  # Confidence score

        # Predict emotion
        expected_shape = emotion_model.input_shape
        face_resized_emotion = cv2.resize(face_roi, (expected_shape[1], expected_shape[2]))
        face_resized_emotion = face_resized_emotion.astype("float32") / 255.0
        face_resized_emotion = np.expand_dims(face_resized_emotion, axis=0)
        
        emotion_pred = emotion_model.predict(face_resized_emotion)
        emotion_label = emotion_labels[np.argmax(emotion_pred)]

        # Determine gender
        if gender_pred[0] == 1:
            female_count += 1
            image_name = f"female{female_count}.jpg"
            image_path = os.path.join(female_dir, image_name)
            gender = "Female"
        else:
            male_count += 1
            image_name = f"male{male_count}.jpg"
            image_path = os.path.join(male_dir, image_name)
            gender = "Male"

        # Save face image
        cv2.imwrite(image_path, face_roi)
        print(f"Image saved to {image_path}")

        # Log detection details
        with open(log_file_path, mode='a', newline='') as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow([image_name, gender, round(accuracy, 4), emotion_label])

        print(f"Final Detection - Gender: {gender}, Emotion: {emotion_label}, Accuracy: {round(accuracy, 4)}")
    except Exception as e:
        print(f"Error processing face: {e}")
else:
    print("No face detected in the last 10 seconds.")
