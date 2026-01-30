from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os


app = Flask(__name__)

model = YOLO('model/best.pt') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        results = model(file_path)
        prediction = 'fracture' if results[0].boxes.cls.numel() > 0 else 'no-fracture'

        result_image_path = os.path.join('static/uploads', 'result_' + file.filename)
        results[0].save(filename=result_image_path)

        return render_template('result.html', prediction=prediction, image_path='uploads/result_' + file.filename)

if __name__ == '__main__':
    app.run(debug=True)