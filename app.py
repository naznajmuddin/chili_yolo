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
    make_response,
)
from datetime import datetime, timedelta
from io import StringIO
import os
import cv2
import csv
from ultralytics import YOLO

UPLOAD_COUNTER = 0
ANALYSIS_COUNTER = 0

RESET_KEYS = [
    "healthy_count",
    "yellowish_count",
    "overall_health",
    "yellowish_percentage",
    "analysis_count",
]

app = Flask(__name__)
app.secret_key = "123456"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

has_reset = False  # Flag to ensure session is cleared only once


@app.before_request
def reset_session_on_start():
    global has_reset
    if not has_reset:
        session.clear()
        print("Session data has been cleared on app start.")
        has_reset = True


# Load the YOLO model
image_model = YOLO("medium_50_best.pt")  # if image validation
video_model = YOLO("best.pt")  # if video feed

CLASS_COLORS = {
    "Healthy": (0, 255, 0),  # Green
    "Yellowish": (0, 255, 255),  # Yellow
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate_frames():
        camera = cv2.VideoCapture(0)
        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break
                else:
                    # YOLO detection logic
                    results = video_model(frame, stream=True)
                    for result in results:
                        boxes = result.boxes.xyxy  # x1, y1, x2, y2
                        confidences = result.boxes.conf
                        class_ids = result.boxes.cls
                        for box, confidence, class_id in zip(
                            boxes, confidences, class_ids
                        ):
                            x1, y1, x2, y2 = map(int, box)
                            class_name = video_model.names[int(class_id)]
                            color = CLASS_COLORS.get(class_name, (255, 255, 255))
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
            camera.release()

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
        increment_counter("upload")
        return redirect(url_for("analyse", filename=file.filename))


@app.before_request
def setup_cumulative_stats():
    if "cumulative_healthy" not in session:
        session["cumulative_healthy"] = 0
    if "cumulative_yellowish" not in session:
        session["cumulative_yellowish"] = 0
    if "total_screened" not in session:
        session["total_screened"] = 0


@app.route("/analyse/<filename>")
def analyse(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image = cv2.imread(filepath)

    # Perform YOLO object detection
    results = image_model(image)
    healthy_count = 0
    yellowish_count = 0

    for result in results:
        boxes = result.boxes.xyxy  # x1, y1, x2, y2
        confidences = result.boxes.conf
        class_ids = result.boxes.cls
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = image_model.names[int(class_id)]
            if class_name == "Healthy":
                healthy_count += 1
            elif class_name == "Yellowish":
                yellowish_count += 1

            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 5
            )

    increment_counter("analysis")

    # Calculate the percentage of Yellowish
    total_count = healthy_count + yellowish_count
    yellowish_percentage = (
        (yellowish_count / total_count) * 100 if total_count > 0 else 0
    )

    overall_health = "Healthy" if yellowish_percentage <= 5 else "Unhealthy"

    height, width, _ = image.shape

    healthy_text = f"Healthy: {healthy_count}"
    yellowish_text = f"Yellowish: {yellowish_count}"
    overall_health_text = (
        f"Yellowish Leaves: {overall_health} ({yellowish_percentage:.1f}%)"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 5

    # Calculate text positions for Healthy and Yellowish
    healthy_size = cv2.getTextSize(healthy_text, font, font_scale, thickness)[0]
    healthy_x = (width - healthy_size[0]) // 2
    healthy_y = height // 2 - 90

    yellowish_size = cv2.getTextSize(yellowish_text, font, font_scale, thickness)[0]
    yellowish_x = (width - yellowish_size[0]) // 2
    yellowish_y = height // 2 - 30

    overall_health_size = cv2.getTextSize(
        overall_health_text, font, font_scale, thickness
    )[0]
    overall_health_x = (width - overall_health_size[0]) // 2
    overall_health_y = height // 2 + 60

    # Background rectangle dimensions for Healthy
    healthy_bg_x1 = healthy_x - 10
    healthy_bg_y1 = healthy_y - healthy_size[1] - 10
    healthy_bg_x2 = healthy_x + healthy_size[0] + 10
    healthy_bg_y2 = healthy_y + 10

    # Background rectangle dimensions for Yellowish
    yellowish_bg_x1 = yellowish_x - 10
    yellowish_bg_y1 = yellowish_y - yellowish_size[1] - 10
    yellowish_bg_x2 = yellowish_x + yellowish_size[0] + 10
    yellowish_bg_y2 = yellowish_y + 10

    # Background rectangle dimensions for Overall Health
    overall_health_bg_x1 = overall_health_x - 10
    overall_health_bg_y1 = overall_health_y - overall_health_size[1] - 10
    overall_health_bg_x2 = overall_health_x + overall_health_size[0] + 10
    overall_health_bg_y2 = overall_health_y + 10

    cv2.rectangle(
        image,
        (healthy_bg_x1, healthy_bg_y1),
        (healthy_bg_x2, healthy_bg_y2),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        image,
        (yellowish_bg_x1, yellowish_bg_y1),
        (yellowish_bg_x2, yellowish_bg_y2),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        image,
        (overall_health_bg_x1, overall_health_bg_y1),
        (overall_health_bg_x2, overall_health_bg_y2),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        healthy_text,
        (healthy_x, healthy_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    cv2.putText(
        image,
        yellowish_text,
        (yellowish_x, yellowish_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    cv2.putText(
        image,
        overall_health_text,
        (overall_health_x, overall_health_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], f"annotated_{filename}")
    cv2.imwrite(annotated_path, image)

    # Track cumulative results for healthy and unhealthy class
    if overall_health == "Healthy":
        session["cumulative_healthy_classifications"] = (
            session.get("cumulative_healthy_classifications", 0) + 1
        )
    else:
        session["cumulative_unhealthy_classifications"] = (
            session.get("cumulative_unhealthy_classifications", 0) + 1
        )

    # Store counts and health in the session
    session["healthy_count"] = healthy_count
    session["yellowish_count"] = yellowish_count
    session["overall_health"] = overall_health
    session["yellowish_percentage"] = yellowish_percentage
    session["analyzed_image"] = f"annotated_{filename}"
    session["cumulative_healthy"] = session.get("cumulative_healthy", 0) + healthy_count
    session["cumulative_yellowish"] = (
        session.get("cumulative_yellowish", 0) + yellowish_count
    )

    session["total_screened"] += 1

    total_cumulative_count = (
        session["cumulative_healthy"] + session["cumulative_yellowish"]
    )
    total_health_percentage = (
        (session["cumulative_yellowish"] / total_cumulative_count) * 100
        if total_cumulative_count > 0
        else 0
    )
    session["total_health_percentage"] = total_health_percentage

    print("Cumulative Healthy:", session["cumulative_healthy"])
    print("Cumulative Yellowish:", session["cumulative_yellowish"])
    print("Total Cumulative Leaves:", total_cumulative_count)
    print("Total Screened:", session["total_screened"])
    
    result = {
        "image_name": f"annotated_{filename}",
        "healthy_count": healthy_count,
        "yellowish_count": yellowish_count,
        "overall_health": overall_health,
    }

    if "analysis_results" not in session:
        session["analysis_results"] = []
    session["analysis_results"].append(result)
    session.modified = True

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
        "healthy": session.get("healthy_count", 0),
        "yellowish": session.get("yellowish_count", 0),
        "overall_health": session.get("overall_health", "Unknown"),
        "yellowish_percentage": session.get("yellowish_percentage", 0),
    }


@app.before_request
def check_and_reset_stats():
    """Check if it's a new day and reset daily stats."""
    if "last_reset" not in session:
        session["last_reset"] = datetime.now().strftime("%Y-%m-%d")

    last_reset_date = session["last_reset"]
    current_date = datetime.now().strftime("%Y-%m-%d")

    if current_date != last_reset_date:
        # Reset the stats
        session["cumulative_healthy"] = 0
        session["cumulative_yellowish"] = 0
        session["total_screened"] = 0

        for key in RESET_KEYS:
            session[key] = 0 if key != "overall_health" else "Unknown"

        session["last_reset"] = current_date


@app.route("/stats")
def stats():
    counters = get_counters()
    counters["total_health_percentage"] = session.get("total_health_percentage", 0)
    counters["analyzed_image_url"] = url_for(
        "show_image", filename=session.get("analyzed_image", "")
    )
    counters["total_cumulative_healthy"] = session.get("cumulative_healthy", 0)
    counters["total_cumulative_yellowish"] = session.get("cumulative_yellowish", 0)

    counters["cumulative_healthy_classifications"] = session.get(
        "cumulative_healthy_classifications", 0
    )
    counters["cumulative_unhealthy_classifications"] = session.get(
        "cumulative_unhealthy_classifications", 0
    )
    return jsonify(counters)


@app.route("/download_csv")
def download_csv():
    data = session.get("analysis_results", [])

    if not data:
        return "No data available to download.", 400

    csv_output = StringIO()
    writer = csv.writer(csv_output)

    # Write the header
    writer.writerow(
        [
            "No.",
            "Image Name",
            "Healthy Leaf Detected",
            "Yellowish Leaf Detected",
            "Chili Plant Health Condition",
        ]
    )

    # Write all rows
    for i, row in enumerate(data, start=1):
        writer.writerow(
            [
                i,
                row["image_name"],
                row["healthy_count"],
                row["yellowish_count"],
                row["overall_health"],
            ]
        )

    csv_data = csv_output.getvalue()
    csv_output.close()

    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=results.csv"
    response.headers["Content-type"] = "text/csv"

    return response



if __name__ == "__main__":
    app.run(debug=True)
