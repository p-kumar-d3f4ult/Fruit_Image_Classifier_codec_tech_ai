import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf
import os
# Load class names from file
import json
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load the trained model
model = tf.keras.models.load_model('fruit_classifier.h5')

# Set image size as used in training
IMG_SIZE = (64, 64)

# Initialize Tkinter Window
root = tk.Tk()
root.title("🍇 Fruit Classifier - AI Powered 🍍")
root.geometry("600x500")
root.configure(bg="#f0f8ff")
root.resizable(False, False)

# -------------------- Styling --------------------
title_label = tk.Label(root, text="Fruit Image Classifier", font=("Helvetica", 20, "bold"), fg="#4a7abc", bg="#f0f8ff")
title_label.pack(pady=10)

# Frame for image and result
frame = tk.Frame(root, bg="#ffffff", bd=2, relief="ridge")
frame.place(relx=0.5, rely=0.55, anchor="center", width=500, height=300)

# Canvas for image
canvas = tk.Label(frame, bg="white")
canvas.pack(pady=10)

# Label for prediction result
result_label = tk.Label(root, text="Select an image to begin...", font=("Arial", 14), bg="#f0f8ff", fg="#333")
result_label.pack(pady=10)

# -------------------- Functions --------------------

def preprocess_image(img_path):
    img = Image.open(img_path)
    img = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def classify_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    try:
        img_array, original_img = preprocess_image(file_path)
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        confidence = float(np.max(predictions)) * 100

        # Display image
        img_resized = original_img.resize((200, 200))
        tk_img = ImageTk.PhotoImage(img_resized)
        canvas.configure(image=tk_img)
        canvas.image = tk_img

        # Determine label (use class_names if available)
        try:
            predicted_label = class_names[class_idx]
        except IndexError:
            predicted_label = f"Class #{class_idx}"

        result_label.config(
            text=f"🍓 Predicted: {predicted_label} ({confidence:.2f}%)",
            fg="#2e8b57"
        )

    except Exception as e:
        messagebox.showerror("Error", f"Failed to classify image.\n\n{str(e)}")

# -------------------- Button --------------------

select_button = tk.Button(
    root,
    text="📂 Select Fruit Image",
    font=("Arial", 12, "bold"),
    command=classify_image,
    bg="#4a7abc",
    fg="white",
    padx=20,
    pady=10,
    bd=0,
    activebackground="#356aa0"
)
select_button.pack(pady=10)

# -------------------- Run App --------------------
root.mainloop()
