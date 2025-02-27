import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
import joblib

def load_images_and_labels(data_dir, img_size=(64, 64)):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")
    
    images = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(0 if label.lower() == "male" else 1)
    return np.array(images), np.array(labels)
data_dir = "C:\\Users\\visha\\OneDrive\\Desktop\\codeminiproject\\new\\image"

try:
    images, labels = load_images_and_labels(data_dir)
    print(f"Loaded {len(images)} images successfully.")
except Exception as e:
    print(f"Error loading images: {e}")
    exit()
images = images / 255.0

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

input_layer = Input(shape=(64, 64, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
feature_extractor = Model(inputs=input_layer, outputs=x)
features_train = feature_extractor.predict(X_train)
features_test = feature_extractor.predict(X_test)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(features_train, y_train)

accuracy = svm_model.score(features_test, y_test)
print(f"SVM Accuracy: {accuracy:.2f}")

os.makedirs("models", exist_ok=True)

joblib.dump(svm_model, "models/gender_svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
feature_extractor.save("models/feature_extractor.h5")
print("Feature extractor, SVM model, and scaler saved successfully.")
