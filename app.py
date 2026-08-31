"""
FaceAttend - Flask web app
Wraps the existing attendance.py face-recognition logic (OpenCV + FaceNet)
in a Flask front-end so it can be deployed as a web app instead of a
desktop script.

Folder layout expected (same repo you already have, plus this app):

face-recognition-attendance/
├── app.py                 <- this file
├── dataset/                <- your existing student folders (unchanged)
├── attendance.csv          <- existing attendance log (unchanged, auto-updated)
├── requirements.txt
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   ├── uploads/            <- group photos user uploads (auto-created)
│   └── results/            <- annotated result images (auto-created)
└── templates/
    ├── base.html
    ├── index.html
    ├── roll_numbers.html
    └── attendance_csv.html
"""

import os
import csv
import uuid
from datetime import datetime

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_file, flash
)

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
THRESHOLD = 0.9  # same threshold used in attendance.py

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-secret-key"  # replace before deploying
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# Roll number mapping - folder name (lowercase) -> roll number.
# Keep this in sync with your dataset/ subfolder names.
ROLL_NUMBERS = {
    "anika": "1",
    "mansi": "2",
    "shiwani": "3",
}

# ---------------------------------------------------------------
# LOAD MODEL + DATASET ONCE AT STARTUP
# (FaceNet is heavy to load, so we do it a single time, not per request)
# ---------------------------------------------------------------
print("Loading FaceNet model...")
from keras_facenet import FaceNet  # noqa: E402  (import after config on purpose)

embedder = FaceNet()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

known_embeddings = []
known_names = []


def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(face_img)[0]
    return embedding


def load_dataset():
    """Build face embeddings for every student image inside dataset/."""
    global known_embeddings, known_names
    known_embeddings = []
    known_names = []

    if not os.path.exists(DATASET_PATH):
        print("Dataset folder not found:", DATASET_PATH)
        return

    for person_name in sorted(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_path):
            continue
        if person_name not in ROLL_NUMBERS:
            print(f"Skipping {person_name}: no roll number mapped.")
            continue

        for file in os.listdir(person_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(person_path, file))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(faces) == 0:
                continue
            x, y, w, h = faces[0]
            face = img[y:y + h, x:x + w]
            try:
                embedding = get_embedding(face)
                known_embeddings.append(embedding)
                known_names.append(person_name)
            except Exception as e:
                print(f"Embedding failed for {file}: {e}")

    print(f"Dataset loaded: {len(set(known_names))} students, "
          f"{len(known_embeddings)} embeddings.")


load_dataset()


# ---------------------------------------------------------------
# ATTENDANCE CSV HELPERS
# ---------------------------------------------------------------
def read_todays_existing_roll_numbers(today):
    existing = set()
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Date") == today and row.get("Roll Number"):
                        existing.add(row["Roll Number"])
        except Exception as e:
            print("Could not read attendance.csv:", e)
    return existing


def append_attendance(rows):
    file_empty = (not os.path.exists(ATTENDANCE_FILE)
                  or os.path.getsize(ATTENDANCE_FILE) == 0)
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if file_empty:
            writer.writerow(["Date", "Time", "Roll Number", "Name",
                              "Status", "Confidence", "Distance", "Method"])
        writer.writerows(rows)


def read_all_attendance():
    rows = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    return rows


def get_roll_number_table():
    students = sorted(set(known_names))
    return [
        {"roll": ROLL_NUMBERS.get(name, "N/A"), "name": name}
        for name in students
    ]


# ---------------------------------------------------------------
# CORE RECOGNITION (mirrors attendance.py, but returns data instead
# of using cv2.imshow, so it works headless on a server)
# ---------------------------------------------------------------
def run_attendance(image_path):
    group = cv2.imread(image_path)
    if group is None:
        return None, "Could not read the uploaded image."

    gray = cv2.cvtColor(group, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    present = set()
    recognition_data = {}
    unknown_count = 0

    for (x, y, w, h) in faces:
        face = group[y:y + h, x:x + w]
        try:
            embedding = get_embedding(face)
        except Exception:
            continue

        if not known_embeddings:
            continue

        distances = [np.linalg.norm(embedding - k) for k in known_embeddings]
        min_idx = int(np.argmin(distances))
        min_distance = distances[min_idx]

        if min_distance < THRESHOLD:
            name = known_names[min_idx]
            present.add(name)
            confidence = max(0, min(100, (1 - min_distance) * 100))
            recognition_data[name] = {
                "confidence": confidence,
                "distance": min_distance,
            }
            cv2.rectangle(group, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(group, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            unknown_count += 1
            cv2.rectangle(group, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(group, "Unknown", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save annotated result image
    result_filename = f"result_{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, group)

    # ---- Save to attendance.csv (skip students already marked today) ----
    today = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%I:%M:%S %p")
    existing_today = read_todays_existing_roll_numbers(today)

    new_rows = []
    marked_today = []  # for display in "just marked" table
    students = sorted(set(known_names))

    for name in students:
        roll_number = ROLL_NUMBERS.get(name, "N/A")
        if roll_number in existing_today:
            continue  # already marked earlier today, don't duplicate

        if name in present:
            status = "Present"
            confidence = f"{recognition_data[name]['confidence']:.2f}%"
            distance = f"{recognition_data[name]['distance']:.3f}"
            method = "Face Recognition"
            time_value = current_time
        else:
            status = "Absent"
            confidence = "-"
            distance = "-"
            method = "-"
            time_value = "-"

        new_rows.append([today, time_value, roll_number, name,
                          status, confidence, distance, method])
        marked_today.append({
            "roll": roll_number, "name": name, "status": status,
            "date": today, "time": time_value,
        })

    if new_rows:
        append_attendance(new_rows)

    summary = {
        "present": len(present),
        "absent": len(set(known_names)) - len(present),
        "total_students": len(set(known_names)),
        "date": today,
        "time": current_time,
        "threshold": THRESHOLD,
        "unknown_detected": unknown_count,
    }

    result = {
        "result_image": url_for("static", filename=f"results/{result_filename}"),
        "summary": summary,
        "marked_today": marked_today,
    }
    return result, None


# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        roll_numbers=get_roll_number_table(),
        result=None,
    )


@app.route("/take-attendance", methods=["POST"])
def take_attendance():
    file = request.files.get("group_image")
    if not file or file.filename == "":
        flash("Please choose a group image to upload.")
        return redirect(url_for("home"))

    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png"):
        flash("Please upload a JPG or PNG image.")
        return redirect(url_for("home"))

    filename = f"upload_{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    result, error = run_attendance(save_path)
    if error:
        flash(error)
        return redirect(url_for("home"))

    return render_template(
        "index.html",
        roll_numbers=get_roll_number_table(),
        result=result,
    )


@app.route("/roll-numbers")
def roll_numbers_page():
    return render_template("roll_numbers.html", roll_numbers=get_roll_number_table())


@app.route("/attendance-csv")
def attendance_csv_page():
    return render_template("attendance_csv.html", rows=read_all_attendance())


@app.route("/attendance-csv/download")
def download_attendance_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        flash("No attendance.csv found yet.")
        return redirect(url_for("attendance_csv_page"))
    return send_file(ATTENDANCE_FILE, as_attachment=True, download_name="attendance.csv")


if __name__ == "__main__":
    # debug=False in production; set host/port via env vars for deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
