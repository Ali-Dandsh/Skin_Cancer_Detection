# 🩺 Skin Cancer Detection (CNN + Flask Web App)

## 📌 Overview
This project is a **Deep Learning-based web application** designed to classify skin lesions (benign vs malignant) using a **Convolutional Neural Network (CNN)** model trained with TensorFlow/Keras.  
It consists of:
- **Model Training**: `skin_cancer_project.py` trains the CNN and saves the model (`skin2.h5`).
- **Web Interface**: `app.py` provides a Flask-based interface for users to upload images and get predictions.

---

## 🚀 Features
- **Upload images** for real-time classification.
- **Deep Learning CNN Model** trained on medical image datasets.
- **Flask Web App** for user interaction.
- **Visualization** of training/validation performance.
- **Confidence score** for predictions.

---

## 🗂 Project Structure
```
Skin_Cancer_Detection/
│
├── app.py                   # Flask application
├── skin_cancer_project.py   # Model training script
├── skin2.h5                 # Trained CNN model
├── static/
│   ├── styles.css           # CSS styles for the web app
│   └── uploads/             # Uploaded images
├── templates/
│   └── index.html           # Web app template
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/Ali-Dandsh/Skin_Cancer_Detection.git
cd Skin_Cancer_Detection
```

### **2. Create a virtual environment**
```bash
python -m venv venv
# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### **Run the Flask App**
```bash
python app.py
```
- Open your browser at: `http://127.0.0.1:5000`
- Upload a skin lesion image to get a prediction (Benign or Malignant).

---

## 🧠 Model Details
- **Architecture:** CNN with Conv2D, MaxPooling, Dropout, and Dense layers.
- **Input Size:** 224x224 RGB images.
- **Framework:** TensorFlow/Keras.
- **Training Script:** The `skin_cancer_project.py` script trains the CNN and saves the model.

---

## 📊 Training Performance
- **Loss & Accuracy:** Visualized during training.
- **Evaluation:** `classification_report` and confusion matrix are generated to assess the model.

---

## 📜 Code Snippets

### **Flask App (app.py)**

```python
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
model = load_model("skin2.h5")

CLASS_NAMES = ['benign', 'malignant']

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = Image.open(filepath).resize((224, 224)).convert("RGB")
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0]
            pred_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100

            return render_template("index.html",
                                   prediction=pred_class,
                                   confidence=confidence,
                                   image_path=filepath)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
```

---

### **Model Training (skin_cancer_project.py)**

```python
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from sklearn.metrics import classification_report

# Dataset path
train_dir = "Skin_Data/melanoma_cancer_dataset/train"
skin = os.listdir(train_dir)
print(skin)

# Count images
nums_train = {}
for s in skin:
    nums_train[s] = len(os.listdir(train_dir + '/' + s))
img_per_class_train = pd.DataFrame(nums_train.values(), index=nums_train.keys(), columns=["no. of images"])
print('Train data distribution :')
print(img_per_class_train)

# Data generators
train_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.25
)
valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.25)

train_data = train_gen.flow_from_directory(
    train_dir, subset='training', target_size=(224,224),
    batch_size=64, color_mode='rgb', class_mode='categorical', shuffle=True
)
test_data = valid_gen.flow_from_directory(
    train_dir, subset='validation', target_size=(224,224),
    batch_size=64, color_mode='rgb', class_mode='categorical', shuffle=False
)

# CNN Model
model_1 = keras.models.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    keras.layers.Dropout(0.1),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.Dropout(0.15),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_1.summary()

# Train
history = model_1.fit(train_data, validation_data=test_data, epochs=15)

# Save model
model_1.save("skin2.h5")

# Evaluate
test_loss, test_acc = model_1.evaluate(test_data)
print('Test accuracy:', test_acc)

# Predictions
y_pred = model_1.predict(test_data)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_data.classes
print(classification_report(y_true, y_pred))
```

---

## 📦 Requirements
```
Flask
tensorflow
numpy
Pillow
Werkzeug
matplotlib
pandas
seaborn
torch
torchvision
scikit-learn
```

---

## 🔗 Links
- **GitHub Repository:** [Skin_Cancer_Detection](https://github.com/Ali-Dandsh/Skin_Cancer_Detection)

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the project, feel free to fork the repo and submit a pull request.

---

## 📜 License
This project is licensed under the **MIT License**.
