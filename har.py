import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from prediction import test_predict  # Import the test_predict function from prediction.py

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Activity Recognition App")
        self.root.geometry("400x400")
        self.root.resizable(False, False)

        # Create upload image button
        self.upload_image_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_image_button.pack(pady=20)

        # Create label to display prediction
        self.prediction_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.prediction_label.pack()

        # Create frame to display image
        self.image_frame = tk.Frame(root, width=300, height=300)
        self.image_frame.pack()

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Call prediction function from prediction.py
            prediction = test_predict(file_path)

            # Display prediction result
            self.prediction_label.config(text=f"Prediction: {prediction}")

            # Display image
            image = Image.open(file_path)
            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
