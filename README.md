
# 🎯 Smart Attendace System
### 🚀 AI-Powered Attendance System using Face Recognition

## 🚀 Project Description

**Face Recognition Attendance System** is an **AI and Computer Vision based attendance management system** built using **Python, OpenCV, FaceNet, Keras-FaceNet, and NumPy**.

The system automatically detects and recognizes students from a **group image** and records their attendance without requiring manual roll calls.

It uses **OpenCV Haar Cascade** for face detection and **FaceNet** to generate facial embeddings. These embeddings are compared with registered student embeddings using **Euclidean Distance** to identify the closest matching student.

Once a student is recognized, the system automatically records their **Name, Roll Number, Date, Time, Status, Confidence, Distance, and Recognition Method** in an attendance CSV file.

The project demonstrates the practical application of **Artificial Intelligence, Machine Learning, Computer Vision, Facial Embeddings, Image Processing, and Data Management** to solve a real-world problem.

---

## 🌟 Key Highlights

- 🤖 AI-based face recognition
- 👤 Automatic face detection
- 🧠 FaceNet facial embeddings
- 📸 Group image recognition
- 👥 Multiple student recognition
- 🆔 Roll number support
- ✅ Automatic Present/Absent marking
- 📅 Automatic date and time recording
- 🎯 Face matching distance
- 📊 Recognition confidence score
- 🔄 Duplicate attendance prevention
- ❓ Unknown face detection
- 📄 Automatic CSV attendance generation
- 📂 Folder-based student dataset
- 🖥️ Visual recognition result
- ⚡ Fast automated attendance processing

---
# 🎥 Demo



## 1 Face Recognition

The system detects faces from the group image and compares them with the registered student faces.

<p align="center">
<img screenshots/demo2.png" width="900">
</p>

---

## 2 Unknown Face Detection

If a detected face does not match any registered student within the recognition threshold, the system identifies the person as **Unknown**.

<p align="center">
<img src="screenshots/result2.png" width="900">
</p>

---

## 4️ Attendance CSV


```text
👥 Faces detected: 3

🔍 Minimum distance: 0.052
✅ Matched: anika
📏 Distance: 0.052
🎯 Confidence: 94.80%

🔍 Minimum distance: 0.031
✅ Matched: mansi
📏 Distance: 0.031
🎯 Confidence: 96.90%


| Date | Time | Roll Number | Name | Status | Confidence | Distance | Method |
|---|---|---|---|---|---|---|---|
| 27-08-2026 | 09:02:15 | 2303101 | Anika | Present | 94.82% | 0.052 | Face Recognition |
| 27-08-2026 | 09:02:16 | 2303121 | Mansi | Present | 97.31% | 0.031 | Face Recognition |
| 27-08-2026 | - | 2303145 | Shiwani | Absent | - | - | - |

---

# ✨ Features

- 👤 Face Detection
- 🧠 Face Recognition using FaceNet
- 📸 Group Image Processing
- 👥 Multiple Face Recognition
- 🆔 Student Roll Number Management
- 📅 Automatic Date Recording
- ⏰ Automatic Time Recording
- ✅ Present/Absent Detection
- 🎯 Face Recognition Distance
- 📊 Confidence Score
- ❓ Unknown Face Detection
- 🔄 Duplicate Attendance Prevention
- 📄 CSV Attendance Storage
- 📂 Folder-Based Dataset
- 🖥️ Recognition Result Visualization

---

# 🏗️ System Architecture

```text
                  Student Dataset
                         │
                         ▼
                 Face Detection
                         │
                         ▼
                Face Preprocessing
                         │
                         ▼
                   FaceNet Model
                         │
                         ▼
                 Face Embeddings
                         │
                         │
                         ▼
                  Group Image
                         │
                         ▼
                 Face Detection
                         │
                         ▼
                Generate Embedding
                         │
                         ▼
             Compare Face Embeddings
                         │
                         ▼
                Euclidean Distance
                         │
                         ▼
                 Recognition Check
                         │
              ┌──────────┴──────────┐
              │                     │
         Match Found            No Match
              │                     │
              ▼                     ▼
           Present               Unknown
              │
              ▼
       Attendance Processing
              │
              ▼
         Attendance CSV
```

# ▶️ How to Run

Follow the steps below to run the **Face Recognition Attendance System** on your local machine.

---

## 

Open a terminal and run the python file:

```bash
python app.py

```
# 🚀 Future Development

The current system provides a working foundation for automated attendance using Face Recognition. The following features can be added to make the system more scalable, secure, and suitable for real-world deployment.

---

## 📷 1. Real-Time Webcam Recognition

Replace the current group-image approach with a live webcam.

```text
Webcam
   ↓
Live Face Detection
   ↓
Face Recognition
   ↓
Automatic Attendance

```
# 🛠️ Used Tools & Technologies

## 🤖 AI / Machine Learning

- **FaceNet** – Generates facial embeddings and performs face feature extraction.
- **Keras-FaceNet** – Provides FaceNet integration for the Python application.

---

## 👁️ Computer Vision

- **OpenCV** – Used for face detection, image processing, face cropping, resizing, and displaying recognition results.
- **Haar Cascade Classifier** – Used for detecting faces in images.

---

## 🐍 Programming Language

- **Python** – Main programming language used to develop the complete attendance system.

---

## 🔢 Data Processing

- **NumPy** – Used for numerical operations, embedding processing, and Euclidean distance calculation.

---

## 📄 Data Storage

- **CSV** – Used to store attendance records including:
  - Date
  - Time
  - Roll Number
  - Name
  - Status
  - Confidence
  - Distance
  - Recognition Method

---

## 💻 Development Tools

- **Visual Studio Code** – Used for coding, debugging, and project development.
- **Git** – Used for version control.
- **GitHub** – Used for source-code hosting and project management.

---

## 📦 Python Libraries

```text
tensorflow
keras-facenet
opencv-python
numpy
```
### Technology WorkFlow
              Python
                │
                ▼
             OpenCV
                │
                ▼
        Face Detection
                │
                ▼
            FaceNet
                │
                ▼
       Face Embeddings
                │
                ▼
             NumPy
                │
                ▼
       Face Comparison
                │
                ▼
      Attendance Processing
                │
                ▼
               CSV
                │
                ▼
             GitHub

```
```
# 👩‍💻 About Me

## Mansi Patel

🎓 **Branch:** Computer Science Engineering (CSE)

📧 **Email:** mansi.patel.23031@iitgoa.ac.in

💻 **GitHub ID:** [mansipatel-git](https://github.com/mansipatel-git)
