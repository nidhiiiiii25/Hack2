import cv2
import torch
from PIL import Image
from torchvision import models, transforms
import os
import serial
import time

# Initialize serial communication with Arduino
try:
    ser = serial.Serial('COM8', 9600, timeout=1)  # Replace 'COM3' with your Arduino's port
    time.sleep(2)  # Wait for serial connection to stabilize
    print("Serial connection established with Arduino")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()

# Verify model file exists
model_path = r'C:\hackethon\pain_detector.pth'
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure 'pain_detector.pth' exists. You may need to re-run 'train.py'.")
    ser.close()
    exit()

# Load model
try:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.to('cpu')
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    ser.close()
    exit()

# Define transform
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load face detector
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"Error: Cascade file not found at {cascade_path}")
    ser.close()
    exit()
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("Error: Could not load face cascade")
    ser.close()
    exit()

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Müller: Webcam still not accessible")
        ser.close()
        exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame")
        break
    # Resize frame and convert to grayscale for face detection
    gray = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
    
    # Process only the first detected face
    output = 0  # Default output: no pain
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle
        if w > 0 and h > 0:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                print("Empty face region, outputting default...")
            else:
                # Preprocess face for model
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = Image.fromarray(face)
                face = transform(face).unsqueeze(0).to('cpu')
                # Run model
                with torch.no_grad():
                    output_tensor = model(face)
                    pred = torch.softmax(output_tensor, dim=1).argmax().item()
                    output = 1 if pred == 1 else 0  # 1 for pain, 0 for no pain
        else:
            print("Invalid face dimensions, outputting default...")
    
    # Send output to Arduino
    try:
        ser.write(str(output).encode())  # Send '0' or '1' as a byte
        print(f"Sent to Arduino: {output}")
    except Exception as e:
        print(f"Error sending to Arduino: {e}")
    
    # Display output on frame (optional, for debugging)
    cv2.putText(frame, f"Pain: {output}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Pain Detection', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()