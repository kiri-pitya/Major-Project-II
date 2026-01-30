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

       
        results = model(file_path)[0]  

       
        boxes = results.boxes
        has_detection = len(boxes) > 0

        if has_detection:
            
            cls_ids = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            
            class_names = results.names  # e.g., {0: 'fracture', 1: 'no-fracture'}
            
            
            max_idx = confs.argmax()
            pred_class_id = int(cls_ids[max_idx])
            pred_class_name = class_names[pred_class_id]
            confidence = float(confs[max_idx])

            prediction = pred_class_name.capitalize()
            confidence_text = f"{confidence:.2%}"
        else:
            prediction = "No Fracture"
            confidence_text = "N/A"

        
        result_image_path = os.path.join('static/uploads', 'result_' + file.filename)
        results.save(filename=result_image_path)

        return render_template(
            'result.html',
            prediction=prediction,
            confidence=confidence_text if has_detection else None,
            image_path='uploads/result_' + file.filename,
            original_image='uploads/' + file.filename  
        )

if __name__ == '__main__':
    app.run(debug=True)