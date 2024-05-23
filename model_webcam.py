import cv2
from tensorflow import keras
import numpy as np
# Load the pre-trained model
model = keras.models.load_model(r'C:\Users\pradhyu\Tkinter_GUI\Har_model_efficient.h5')

# Function to resize the frame
def resize_frame(frame, target_size=(160, 160)):
    return cv2.resize(frame, target_size)


labels = ['sitting', 'using_laptop', 'hugging', 'sleeping', 'drinking',
       'clapping', 'dancing', 'cycling', 'calling', 'laughing', 'eating',
       'fighting', 'listening_to_music', 'running', 'texting', 'smoking',
       'weapon', 'knife']

# Function to draw label on the frame
def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera was opened correctly
if not cap.isOpened():
    print("Could not open video device")

# Main loop to capture frames
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    
    if not ret:
        print("Error: Could not read frame")
        break
    
    resized_frame = resize_frame(frame)  # Resize the frame
    resized_frame = resized_frame / 255.0  # Normalize the frame
    
    # Predict using the model
    prediction = model.predict(resized_frame[np.newaxis, ...], steps=1)
    # print("npmax result: ",np.max(prediction))
    itemindex = np.where(prediction==np.max(prediction))
    # print("Itemindex: ",itemindex)
    prediction = itemindex[1][0]
    print("prediction:",prediction)
    print("probability: "+str(np.max(prediction)*100) + "%\nPredicted class : ", prediction)
    # print("Lable: ",labels[prediction])
    
    # Draw label on the frame
    draw_label(frame, 'Label: {}'.format(labels[prediction]), (20, 20), (255, 0, 0))
    
    # Display the frame
    cv2.imshow("preview", frame)
    
    # Wait for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
