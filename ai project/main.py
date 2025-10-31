from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Set correct file paths
HAAR_CASCADE_PATH = r'C:\Users\ayush\Desktop\ai project\haarcascade_frontalface_default.xml'
MODEL_PATH = r'C:\Users\ayush\Desktop\ai project\model.h5'

# Load Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Load the trained emotion classification model
classifier = load_model(MODEL_PATH)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)  # Shape: (1, 48, 48, 1)

            # Make prediction
            prediction = classifier.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)

            # Put text on the image
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
