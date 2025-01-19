from flask import Flask, Response, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory to store uploaded images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with your model if different

@app.route('/')
def index():
    return render_template('index.html')  # HTML template for the main interface

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        camera = cv2.VideoCapture(0)  # Open webcam
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # YOLO detection logic
                results = model(frame, stream=True)
                for result in results:
                    boxes = result.boxes.xyxy  # x1, y1, x2, y2
                    confidences = result.boxes.conf
                    class_ids = result.boxes.cls
                    for box, confidence, class_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        label = f"{model.names[int(class_id)]} {confidence:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('analyse', filename=file.filename))

@app.route('/analyse/<filename>')
def analyse(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)

    # Perform YOLO object detection
    results = model(image)
    for result in results:
        boxes = result.boxes.xyxy  # x1, y1, x2, y2
        confidences = result.boxes.conf
        class_ids = result.boxes.cls
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(class_id)]} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image
    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], f'annotated_{filename}')
    cv2.imwrite(annotated_path, image)

    # Redirect directly to the image
    return redirect(url_for('show_image', filename=f'annotated_{filename}'), code=302)



@app.route('/show_image/<filename>')
def show_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
