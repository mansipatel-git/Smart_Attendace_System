"""
Smart Attendance System
Flask + OpenCV LBPH Face Recognition

Lightweight version for free deployment.
No TensorFlow / FaceNet required.
"""

import os
import csv
import uuid
from datetime import datetime

import cv2
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


app = Flask(__name__)

# Use environment variable in deployment.
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "smart-attendance-development-key"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# ROLL NUMBER MAPPING
# ============================================================

# Keep these names exactly the same as your dataset folders.

ROLL_NUMBERS = {
    "anika": "1",
    "mansi": "2",
    "shiwani": "3",
}


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade face detector.")


# ============================================================
# LBPH FACE RECOGNIZER
# ============================================================

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "cv2.face is not available. "
        "Install opencv-contrib-python-headless."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)


# Maps numeric LBPH labels to student names.
label_to_name = {}

# Students that actually have usable training images.
known_names = set()

# LBPH confidence:
# LOWER value = better match.
#
# 50-70 -> usually stronger match
# 70-90 -> moderate
# > 90  -> usually unknown
LBPH_THRESHOLD = 85.0


# ============================================================
# FACE PREPROCESSING
# ============================================================

def prepare_face(face):
    """
    Convert a face to grayscale and resize it to a fixed size.
    """

    if face is None or face.size == 0:
        return None

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        (160, 160),
        interpolation=cv2.INTER_AREA
    )

    # Improve contrast slightly.
    gray = cv2.equalizeHist(gray)

    return gray


# ============================================================
# DETECT FACE FROM TRAINING IMAGE
# ============================================================

def extract_training_faces(image):
    """
    Detect faces from a dataset image.
    Returns prepared grayscale face images.
    """

    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    result = []

    for x, y, w, h in faces:

        face = image[
            y:y + h,
            x:x + w
        ]

        prepared = prepare_face(face)

        if prepared is not None:
            result.append(prepared)

    return result


# ============================================================
# TRAIN LBPH MODEL
# ============================================================

def train_recognizer():

    global label_to_name
    global known_names

    training_faces = []
    training_labels = []

    label_to_name = {}
    known_names = set()

    if not os.path.exists(DATASET_PATH):

        print("Dataset folder not found:")
        print(DATASET_PATH)

        return

    label_id = 0

    print("========================================")
    print("Training lightweight LBPH recognizer...")
    print("========================================")

    for person_name in sorted(os.listdir(DATASET_PATH)):

        person_path = os.path.join(
            DATASET_PATH,
            person_name
        )

        if not os.path.isdir(person_path):
            continue

        person_name = person_name.lower().strip()

        # Only students present in ROLL_NUMBERS.
        if person_name not in ROLL_NUMBERS:

            print(
                f"Skipping {person_name}: "
                "no roll number mapping."
            )

            continue

        person_face_count = 0

        for filename in sorted(os.listdir(person_path)):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                person_path,
                filename
            )

            image = cv2.imread(image_path)

            if image is None:

                print(
                    f"Could not read: {image_path}"
                )

                continue

            faces = extract_training_faces(image)

            if not faces:
                print(
                    f"No face found in: "
                    f"{person_name}/{filename}"
                )

                continue

            # Use all detected faces from the training image.
            for face in faces:

                training_faces.append(face)
                training_labels.append(label_id)

                person_face_count += 1

        if person_face_count > 0:

            label_to_name[label_id] = person_name
            known_names.add(person_name)

            print(
                f"{person_name}: "
                f"{person_face_count} training face(s)"
            )

            label_id += 1

    if not training_faces:

        print("ERROR: No training faces found.")
        return

    print("----------------------------------------")
    print(
        f"Training with {len(training_faces)} face images..."
    )

    recognizer.train(
        training_faces,
        __import__("numpy").array(
            training_labels,
            dtype="int32"
        )
    )

    print(
        f"LBPH training complete. "
        f"Students: {len(known_names)}"
    )

    print("========================================")


# Train once when the application starts.
train_recognizer()


# ============================================================
# ATTENDANCE CSV FUNCTIONS
# ============================================================

def read_todays_existing_roll_numbers(today):

    existing = set()

    if not os.path.exists(ATTENDANCE_FILE):
        return existing

    try:

        with open(
            ATTENDANCE_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                if (
                    row.get("Date") == today
                    and row.get("Roll Number")
                ):

                    existing.add(
                        row["Roll Number"]
                    )

    except Exception as e:

        print(
            "Could not read attendance.csv:",
            e
        )

    return existing


def append_attendance(rows):

    file_empty = (
        not os.path.exists(ATTENDANCE_FILE)
        or os.path.getsize(ATTENDANCE_FILE) == 0
    )

    with open(
        ATTENDANCE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if file_empty:

            writer.writerow([
                "Date",
                "Time",
                "Roll Number",
                "Name",
                "Status",
                "Confidence",
                "Distance",
                "Method"
            ])

        writer.writerows(rows)


def read_all_attendance():

    if not os.path.exists(ATTENDANCE_FILE):
        return []

    try:

        with open(
            ATTENDANCE_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            return list(reader)

    except Exception as e:

        print(
            "Could not read attendance.csv:",
            e
        )

        return []


def get_roll_number_table():

    students = sorted(known_names)

    return [
        {
            "roll": ROLL_NUMBERS.get(
                name,
                "N/A"
            ),
            "name": name
        }
        for name in students
    ]


# ============================================================
# CONFIDENCE CONVERSION
# ============================================================

def calculate_confidence(distance):

    """
    LBPH returns a distance/confidence value.

    Lower distance = better match.

    Convert it into an approximate percentage
    for displaying in the UI.
    """

    confidence = 100.0 - (
        (distance / 100.0) * 100.0
    )

    confidence = max(
        0.0,
        min(100.0, confidence)
    )

    return confidence


# ============================================================
# FACE RECOGNITION
# ============================================================

def run_attendance(image_path):

    group = cv2.imread(image_path)

    if group is None:

        return (
            None,
            "Could not read the uploaded image."
        )

    if not known_names:

        return (
            None,
            "No students were found in the dataset."
        )

    # --------------------------------------------
    # Detect faces in group image
    # --------------------------------------------

    gray = cv2.cvtColor(
        group,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    present = set()
    recognition_data = {}

    unknown_count = 0

    # --------------------------------------------
    # Recognize each detected face
    # --------------------------------------------

    for (x, y, w, h) in faces:

        face = group[
            y:y + h,
            x:x + w
        ]

        prepared_face = prepare_face(face)

        if prepared_face is None:
            continue

        try:

            label, distance = recognizer.predict(
                prepared_face
            )

        except Exception as e:

            print(
                "Recognition error:",
                e
            )

            continue

        name = label_to_name.get(
            int(label)
        )

        # ----------------------------------------
        # Known face
        # ----------------------------------------

        if (
            name is not None
            and distance <= LBPH_THRESHOLD
        ):

            present.add(name)

            confidence = calculate_confidence(
                distance
            )

            recognition_data[name] = {
                "confidence": confidence,
                "distance": distance
            }

            # Green rectangle
            cv2.rectangle(
                group,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                group,
                name,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # ----------------------------------------
        # Unknown face
        # ----------------------------------------

        else:

            unknown_count += 1

            cv2.rectangle(
                group,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                group,
                "Unknown",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # ========================================================
    # SAVE RESULT IMAGE
    # ========================================================

    result_filename = (
        f"result_{uuid.uuid4().hex}.jpg"
    )

    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )

    cv2.imwrite(
        result_path,
        group
    )

    # ========================================================
    # ATTENDANCE
    # ========================================================

    today = datetime.now().strftime(
        "%d-%m-%Y"
    )

    current_time = datetime.now().strftime(
        "%I:%M:%S %p"
    )

    existing_today = (
        read_todays_existing_roll_numbers(
            today
        )
    )

    new_rows = []
    marked_today = []

    students = sorted(known_names)

    for name in students:

        roll_number = ROLL_NUMBERS.get(
            name,
            "N/A"
        )

        # Already marked today.
        if roll_number in existing_today:
            continue

        # ----------------------------------------
        # Present
        # ----------------------------------------

        if name in present:

            status = "Present"

            confidence = (
                f"{recognition_data[name]['confidence']:.2f}%"
            )

            distance = (
                f"{recognition_data[name]['distance']:.3f}"
            )

            method = "LBPH Face Recognition"

            time_value = current_time

        # ----------------------------------------
        # Absent
        # ----------------------------------------

        else:

            status = "Absent"

            confidence = "-"
            distance = "-"
            method = "-"
            time_value = "-"

        new_rows.append([
            today,
            time_value,
            roll_number,
            name,
            status,
            confidence,
            distance,
            method
        ])

        marked_today.append({
            "roll": roll_number,
            "name": name,
            "status": status,
            "date": today,
            "time": time_value
        })

    if new_rows:
        append_attendance(new_rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    total_students = len(
        known_names
    )

    summary = {

        "present": len(present),

        "absent": (
            total_students
            - len(present)
        ),

        "total_students": total_students,

        "date": today,

        "time": current_time,

        "threshold": LBPH_THRESHOLD,

        "unknown_detected": unknown_count
    }

    result = {

        "result_image": url_for(
            "static",
            filename=(
                f"results/{result_filename}"
            )
        ),

        "summary": summary,

        "marked_today": marked_today
    }

    return result, None


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html",
        roll_numbers=get_roll_number_table(),
        result=None
    )


# ============================================================
# TAKE ATTENDANCE
# ============================================================

@app.route(
    "/take-attendance",
    methods=["POST"]
)
def take_attendance():

    file = request.files.get(
        "group_image"
    )

    if (
        not file
        or file.filename == ""
    ):

        flash(
            "Please choose a group image to upload."
        )

        return redirect(
            url_for("home")
        )

    ext = os.path.splitext(
        file.filename
    )[1].lower()

    if ext not in (
        ".jpg",
        ".jpeg",
        ".png"
    ):

        flash(
            "Please upload a JPG or PNG image."
        )

        return redirect(
            url_for("home")
        )

    filename = (
        f"upload_{uuid.uuid4().hex}{ext}"
    )

    save_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(save_path)

    result, error = run_attendance(
        save_path
    )

    if error:

        flash(error)

        return redirect(
            url_for("home")
        )

    return render_template(
        "index.html",
        roll_numbers=get_roll_number_table(),
        result=result
    )


# ============================================================
# ROLL NUMBERS
# ============================================================

@app.route("/roll-numbers")
def roll_numbers_page():

    return render_template(
        "roll_numbers.html",
        roll_numbers=get_roll_number_table()
    )


# ============================================================
# ATTENDANCE CSV PAGE
# ============================================================

@app.route("/attendance-csv")
def attendance_csv_page():

    return render_template(
        "attendance_csv.html",
        rows=read_all_attendance()
    )


# ============================================================
# DOWNLOAD CSV
# ============================================================

@app.route(
    "/attendance-csv/download"
)
def download_attendance_csv():

    if not os.path.exists(
        ATTENDANCE_FILE
    ):

        flash(
            "No attendance.csv found yet."
        )

        return redirect(
            url_for("attendance_csv_page")
        )

    return send_file(
        ATTENDANCE_FILE,
        as_attachment=True,
        download_name="attendance.csv"
    )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
