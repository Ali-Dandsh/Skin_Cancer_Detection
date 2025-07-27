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
