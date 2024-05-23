import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

# Load your pre-trained action recognition model
model_path = "mobilenetv3large.h5"
model = load_model(model_path)

# Define the labels for different actions
labels = ['sitting', 'using_laptop', 'hugging', 'sleeping', 'drinking',
          'clapping', 'dancing', 'cycling', 'calling', 'laughing', 'eating',
          'fighting', 'listening_to_music', 'running', 'texting', 'smoking',
          'weapon', 'knife']

# Open a video capture object (replace '0' with your video source index or path)
cap = cv2.VideoCapture("invideo-ai.mp4")
# cap = cv2.VideoCapture("invideo-ai-1080 The Spectrum of Human Activities 2024-04-30.mp4")
def read_img(fn):
    img = Image.open(fn)
    return np.asarray(img.resize((160,160)))
def resize_window(frame, window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create resizable window
    cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])  # Resize window


while cap.isOpened():
    ret, frame = cap.read()
    # print(frame.shape) #(480, 640, 3)
    if not ret:
        break
    
#     # Resize the frame to match the input size expected by the model
    input_img = cv2.resize(frame, (160, 160))
    # input_img = np.asarray(frame.resize((160,160)))
    
#     # Preprocess the input image for the model
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
    # print(input_img.shape)
    # input_img = input_img / 255.0  # Normalize pixel values
    
    # Predict action probabilities for the input image
    result = model.predict(input_img)
    # result = model.predict(np.asarray([read_img(frame)]))
    # print(result)
    print("score :: ",np.max(result))
    # Check if the predicted action confidence is above a threshold (e.g., 0.6)
    if np.max(result) > 0.7:
        prediction_idx = np.argmax(result)  # Get the index of the predicted action
        predicted_label = labels[prediction_idx]  # Get the predicted label
        
        # Display the predicted action label and bounding box on the frame
        cv2.putText(frame, predicted_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 3)  # Draw a bounding box
        # resize_window(frame, 'Action Recognition')
        imS = cv2.resize(frame, (860, 540)) 
        # Show the frame with bounding box and predicted label
        cv2.imshow('Action Recognition', imS)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
