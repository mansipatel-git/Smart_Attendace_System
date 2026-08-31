import cv2
import numpy as np
import os
import csv
import builtins
from datetime import datetime
from keras_facenet import FaceNet


# ============================================================
# 1. INITIALIZE FACENET
# ============================================================

print("🔄 Loading FaceNet model...")

embedder = FaceNet()

builtins.print("✅ FaceNet model loaded!")


# ============================================================
# 2. FUNCTION: GET FACE EMBEDDING
# ============================================================

def get_embedding(face):

    # FaceNet requires 160 x 160 image
    face = cv2.resize(face, (160, 160))

    # Add batch dimension
    face = np.expand_dims(face, axis=0)

    # Generate embedding
    embedding = embedder.embeddings(face)[0]

    return embedding


# ============================================================
# 3. FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# 4. DATASET PATH
# ============================================================

dataset_path = "dataset"


if not os.path.exists(dataset_path):

    print("❌ Dataset folder not found!")

    print(
        "Expected folder:",
        dataset_path
    )

    exit()


# ============================================================
# 5. ROLL NUMBER MAPPING
# ============================================================

# IMPORTANT:
# Folder name must match the key here.

roll_numbers = {

    "anika": "1",

    "mansi": "2",

    "shiwani": "3"

}


# ============================================================
# 6. LOAD DATASET
# ============================================================

known_embeddings = []

known_names = []


print("\n========================================")
print("📂 LOADING DATASET")
print("========================================")


# Go through every student folder

for person_name in os.listdir(dataset_path):


    person_path = os.path.join(
        dataset_path,
        person_name
    )


    # Ignore files
    if not os.path.isdir(person_path):

        continue


    print(
        f"\n👤 Student: {person_name}"
    )


    # Check roll number

    if person_name not in roll_numbers:

        print(
            f"⚠️ Roll number not found "
            f"for {person_name}"
        )

        print(
            "⚠️ Student skipped!"
        )

        continue


    # --------------------------------------------------------
    # Read images inside student folder
    # --------------------------------------------------------

    image_count = 0


    for file in os.listdir(person_path):


        # Only image files

        if not file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            continue


        image_path = os.path.join(
            person_path,
            file
        )


        # Read image

        img = cv2.imread(
            image_path
        )


        if img is None:

            print(
                f"❌ Failed to load: {file}"
            )

            continue


        # Convert to grayscale

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        # Detect faces

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )


        # No face

        if len(faces) == 0:

            print(
                f"❌ No face found: {file}"
            )

            continue


        # ----------------------------------------------------
        # Take first detected face
        # ----------------------------------------------------

        x, y, w, h = faces[0]


        face = img[
            y:y+h,
            x:x+w
        ]


        try:

            # Generate embedding

            embedding = get_embedding(
                face
            )


            # Store embedding

            known_embeddings.append(
                embedding
            )


            # Store person's name

            known_names.append(
                person_name
            )


            image_count += 1


            print(
                f"✅ Loaded: {file}"
            )


        except Exception as e:

            print(
                f"❌ Embedding failed: {file}"
            )

            print(
                "Error:",
                e
            )


    print(
        f"📸 Images loaded for "
        f"{person_name}: {image_count}"
    )


# ============================================================
# 7. DATASET SUMMARY
# ============================================================

unique_students = sorted(
    set(known_names)
)


print("\n========================================")
print("📊 DATASET SUMMARY")
print("========================================")

print(
    "👥 Total students:",
    len(unique_students)
)

print(
    "📸 Total embeddings:",
    len(known_embeddings)
)

print(
    "👤 Students:",
    unique_students
)


if len(known_embeddings) == 0:

    print(
        "\n❌ No valid dataset images!"
    )

    exit()


print(
    "\n✅ Dataset loaded successfully!"
)


# ============================================================
# 8. LOAD GROUP IMAGE
# ============================================================

group_path = "group.jpeg"


group = cv2.imread(
    group_path
)


if group is None:

    print(
        "\n❌ Group image not found!"
    )

    print(
        "Expected:",
        group_path
    )

    exit()


print("\n========================================")
print("🖼️ PROCESSING GROUP IMAGE")
print("========================================")


# ============================================================
# 9. DETECT FACES IN GROUP IMAGE
# ============================================================

gray = cv2.cvtColor(
    group,
    cv2.COLOR_BGR2GRAY
)


faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)


print(
    f"👥 Faces detected: {len(faces)}"
)


# ============================================================
# 10. RECOGNIZE FACES
# ============================================================

# Set of students who are present

present = set()


# Store recognition information

recognition_data = {}


# Face recognition threshold

THRESHOLD = 0.9


for (x, y, w, h) in faces:


    # --------------------------------------------------------
    # Crop face
    # --------------------------------------------------------

    face = group[
        y:y+h,
        x:x+w
    ]


    try:

        # Generate embedding

        embedding = get_embedding(
            face
        )


    except Exception as e:

        print(
            "❌ Could not generate "
            "face embedding"
        )

        print(
            "Error:",
            e
        )

        continue


    # --------------------------------------------------------
    # Calculate distance
    # --------------------------------------------------------

    distances = [

        np.linalg.norm(
            embedding - known_embedding
        )

        for known_embedding
        in known_embeddings

    ]


    if len(distances) == 0:

        continue


    # Find closest embedding

    min_idx = np.argmin(
        distances
    )


    min_distance = distances[
        min_idx
    ]


    print(
        f"\n🔍 Minimum distance: "
        f"{min_distance:.3f}"
    )


    # ========================================================
    # MATCH FOUND
    # ========================================================

    if min_distance < THRESHOLD:


        name = known_names[
            min_idx
        ]


        # Add student to present set

        present.add(
            name
        )


        # ----------------------------------------------------
        # Calculate display confidence
        # ----------------------------------------------------

        confidence = max(
            0,
            min(
                100,
                (1 - min_distance) * 100
            )
        )


        # ----------------------------------------------------
        # Store recognition data
        # ----------------------------------------------------

        recognition_data[name] = {

            "confidence": confidence,

            "distance": min_distance

        }


        print(
            f"✅ Matched: {name}"
        )


        print(
            f"📏 Distance: "
            f"{min_distance:.3f}"
        )


        print(
            f"🎯 Confidence: "
            f"{confidence:.2f}%"
        )


        # ----------------------------------------------------
        # Draw GREEN rectangle
        # ----------------------------------------------------

        cv2.rectangle(
            group,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Display name
        # ----------------------------------------------------

        cv2.putText(
            group,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # UNKNOWN FACE
    # ========================================================

    else:


        print(
            "❌ Unknown person"
        )


        print(
            f"📏 Distance: "
            f"{min_distance:.3f}"
        )


        # ----------------------------------------------------
        # Draw RED rectangle
        # ----------------------------------------------------

        cv2.rectangle(
            group,
            (x, y),
            (x+w, y+h),
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # Display Unknown
        # ----------------------------------------------------

        cv2.putText(
            group,
            "Unknown",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


# ============================================================
# 11. ATTENDANCE INFORMATION
# ============================================================

attendance_file = "attendance.csv"


# Current date

today = datetime.now().strftime(
    "%d-%m-%Y"
)


# Current time

current_time = datetime.now().strftime(
    "%H:%M:%S"
)


print("\n========================================")
print("📝 SAVING ATTENDANCE")
print("========================================")


# ============================================================
# 12. READ EXISTING ATTENDANCE
# ============================================================

existing_records = set()


if os.path.exists(
    attendance_file
):


    try:

        with open(
            attendance_file,
            "r",
            newline=""
        ) as f:


            reader = csv.DictReader(
                f
            )


            for row in reader:


                date_value = row.get(
                    "Date"
                )


                roll_value = row.get(
                    "Roll Number"
                )


                # Check if attendance already
                # exists for today

                if (
                    date_value == today
                    and roll_value
                ):

                    existing_records.add(
                        roll_value
                    )


    except Exception as e:

        print(
            "⚠️ Could not read "
            "attendance.csv"
        )

        print(
            "Error:",
            e
        )


# ============================================================
# 13. CREATE CSV IF REQUIRED
# ============================================================

file_empty = (

    not os.path.exists(
        attendance_file
    )

    or

    os.path.getsize(
        attendance_file
    ) == 0

)


with open(
    attendance_file,
    "a",
    newline=""
) as f:


    writer = csv.writer(
        f
    )


    # --------------------------------------------------------
    # CSV HEADER
    # --------------------------------------------------------

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


    # ========================================================
    # 14. SAVE EACH STUDENT
    # ========================================================

    students = sorted(
        set(known_names)
    )


    for name in students:


        # ----------------------------------------------------
        # Get roll number
        # ----------------------------------------------------

        roll_number = roll_numbers.get(
            name,
            "N/A"
        )


        # ----------------------------------------------------
        # Check duplicate attendance
        # ----------------------------------------------------

        if roll_number in existing_records:

            print(
                f"⚠️ Already recorded today: "
                f"{roll_number} - {name}"
            )

            continue


        # ====================================================
        # PRESENT
        # ====================================================

        if name in present:


            status = "Present"


            # Get recognition data

            confidence = (

                f"{recognition_data[name]['confidence']:.2f}%"

            )


            distance = (

                f"{recognition_data[name]['distance']:.3f}"

            )


            method = "Face Recognition"


            time_value = current_time


        # ====================================================
        # ABSENT
        # ====================================================

        else:


            status = "Absent"


            confidence = "-"


            distance = "-"


            method = "-"


            time_value = "-"


        # ----------------------------------------------------
        # Write record
        # ----------------------------------------------------

        writer.writerow([

            today,

            time_value,

            roll_number,

            name,

            status,

            confidence,

            distance,

            method

        ])


        print(

            f"✅ {roll_number} | "

            f"{name} | "

            f"{status}"

        )


print(
    "\n✅ Attendance saved successfully!"
)


# ============================================================
# 15. DISPLAY RESULT IMAGE
# ============================================================

cv2.imshow(
    "Face Recognition Attendance",
    group
)


print(
    "\n🖥️ Press any key to close."
)


cv2.waitKey(0)

cv2.destroyAllWindows()


# ============================================================
# 16. FINAL ATTENDANCE SUMMARY
# ============================================================

all_students = set(
    known_names
)


absent = (
    all_students - present
)


print("\n========================================")
print("🎉 ATTENDANCE SUMMARY")
print("========================================")


print(
    f"👥 Total students : "
    f"{len(all_students)}"
)


print(
    f"✅ Present        : "
    f"{len(present)}"
)


print(
    f"❌ Absent         : "
    f"{len(absent)}"
)


# ------------------------------------------------------------
# Present students
# ------------------------------------------------------------

print(
    "\n✅ PRESENT STUDENTS:"
)


for name in sorted(present):


    roll_number = roll_numbers.get(
        name,
        "N/A"
    )


    print(
        f"   {roll_number} - {name}"
    )


# ------------------------------------------------------------
# Absent students
# ------------------------------------------------------------

print(
    "\n❌ ABSENT STUDENTS:"
)


for name in sorted(absent):


    roll_number = roll_numbers.get(
        name,
        "N/A"
    )


    print(
        f"   {roll_number} - {name}"
    )


print(
    "\n========================================"
)

print(
    "🎯 Program completed successfully!"
)

print(
    "========================================"
)