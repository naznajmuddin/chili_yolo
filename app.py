from flask import (
    Flask,
    Response,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    session,
    jsonify,
)
import os
import cv2
from ultralytics import YOLO

UPLOAD_COUNTER = 0
ANALYSIS_COUNTER = 0

app = Flask(__name__)
app.secret_key = "123456"
app.config["UPLOAD_FOLDER"] = "uploads"  # Directory to store uploaded images
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the YOLO model
model = YOLO("best_v8large.pt")  # Replace 'yolov8n.pt' with your model if different

# Define a color map for the classes
CLASS_COLORS = {
    "Healthy": (0, 255, 0),  # Green
    "Yellowish": (0, 255, 255),  # Yellow
}


@app.route("/")
def index():
    return render_template("index.html")  # HTML template for the main interface


@app.route("/video_feed")
def video_feed():
    def generate_frames():
        camera = cv2.VideoCapture(0)  # Open webcam
        try:
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
                        for box, confidence, class_id in zip(
                            boxes, confidences, class_ids
                        ):
                            x1, y1, x2, y2 = map(int, box)
                            class_name = model.names[
                                int(class_id)
                            ]  # Get the class name
                            color = CLASS_COLORS.get(
                                class_name, (255, 255, 255)
                            )  # Default to white if class not in map
                            label = f"{class_name} {confidence:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                color,
                                5,
                            )

                    ret, buffer = cv2.imencode(".jpg", frame)
                    frame = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
        finally:
            camera.release()  # Ensure camera is released when loop exits

    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        increment_counter("upload")  # Increment the upload counter here
        return redirect(url_for("analyse", filename=file.filename))


@app.route("/analyse/<filename>")
def analyse(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image = cv2.imread(filepath)

    # Perform YOLO object detection
    results = model(image)
    for result in results:
        boxes = result.boxes.xyxy  # x1, y1, x2, y2
        confidences = result.boxes.conf
        class_ids = result.boxes.cls
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(class_id)]  # Get the class name
            color = CLASS_COLORS.get(
                class_name, (255, 255, 255)
            )  # Default to white if class not in map
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 5
            )

    # Increment the analysis counter here
    increment_counter("analysis")

    # Save the annotated image
    annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], f"annotated_{filename}")
    cv2.imwrite(annotated_path, image)

    # Redirect directly to the image
    return redirect(url_for("show_image", filename=f"annotated_{filename}"), code=302)


@app.route("/show_image/<filename>")
def show_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.before_request
def setup_session():
    global UPLOAD_COUNTER, ANALYSIS_COUNTER
    if "upload_count" not in session:
        session["upload_count"] = UPLOAD_COUNTER
    if "analysis_count" not in session:
        session["analysis_count"] = ANALYSIS_COUNTER


def increment_counter(counter_type):
    global UPLOAD_COUNTER, ANALYSIS_COUNTER
    if counter_type == "upload":
        UPLOAD_COUNTER += 1
        session["upload_count"] = UPLOAD_COUNTER
    elif counter_type == "analysis":
        ANALYSIS_COUNTER += 1
        session["analysis_count"] = ANALYSIS_COUNTER


def get_counters():
    return {
        "uploaded": session.get("upload_count", 0),
        "analyzed": session.get("analysis_count", 0),
    }


@app.route("/stats")
def stats():
    counters = get_counters()
    return jsonify(counters)


if __name__ == "__main__":
    app.run(debug=True)
