# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 06:53:12 2025

@author: fjwu
"""

import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn

# Initialize OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the pretrained model
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification (real vs. fake)
model.eval()

model_path = "path_to_your_trained_model.pth"
model.load_state_dict(torch.load(model_path))

# Preprocessing transforms for input to the model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Real-time video processing
def process_video():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert frame to grayscale for Haar Cascade
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop and preprocess the face
            face = frame[y:y+h, x:x+w]
            face_tensor = transform(face).unsqueeze(0)  # Add batch dimension

            # Pass through the model
            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)

            # Label the face
            label = "Real" if predicted.item() == 0 else "Fake"
            color = (0, 255, 0) if label == "Real" else (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the frame
        cv2.imshow("Deepfake Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
