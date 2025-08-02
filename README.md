
<img width="1280" alt="readme-banner" src="https://github.com/user-attachments/assets/35332e92-44cb-425b-9dff-27bcf1023c6c">

# MMA Shadow Fighter 🎯

## Basic Details

### Team Name: Team Pazhampori
### Team Members

* Team Lead: Jerry Berna - Albertian Institute of Science and Technology (AISAT)
* Member 2: Suha Shajahan - Albertian Institute of Science and Technology (AISAT)

### Project Description

This project is an AI-powered MMA shadow fighting trainer that uses computer vision to track body movements and classify them as martial arts actions like punch, kick, guard, or grapple. You can also record and mimic your own moves using "shadow mode".

---

### The Problem (that doesn’t exist)

Have you ever wanted to fight yourself in real time? Probably not—but now you can! Bored of shadowboxing alone? Don't worry, we’ve given your shadow an actual memory.

---

### The Solution (that nobody asked for)

Using OpenCV and Mediapipe, we capture your live skeleton movements, classify your fighting style, and even let you record your own fight so you can train against... yourself. You vs. Past You: the ultimate battle.

---

## Technical Details

### Technologies/Components Used

For Software:

* Python
* OpenCV
* Mediapipe
* Unity (for future visualization)
* C# (planned integration)

For Hardware:

* Webcam or laptop camera
* (Optional) ESP32-S3 (for physical arm mimicry)
* Any laptop capable of running OpenCV and Python
* USB cable (if using microcontroller)

---

### Implementation

For Software:

# Installation

Clone the repo and install dependencies:

bash
git clone https://github.com/yourusername/mma-shadow-fighter.git  
cd mma-shadow-fighter  
pip install -r requirements.txt


# Run

bash
python app.py


---

### Project Documentation

# Screenshots

<img width="1916" height="962" alt="Screenshot 2025-08-02 192417" src="https://github.com/user-attachments/assets/8c828390-00eb-499d-9d63-4eb10e872080" />


<img width="1918" height="976" alt="Screenshot 2025-08-02 192955" src="https://github.com/user-attachments/assets/ae0e19e8-d478-43e1-b187-4f1bf822bc68" />


![Screenshot3](screenshots/suggestion_chart.png)
Action label display — tracks what you’re doing.

# Diagrams

![Workflow](diagrams/workflow.png)
Pose detection → Keypoint processing → Action classification → Shadow rendering.

---

# Build Photos

<img width="1920" height="1080" alt="Screenshot (35)" src="https://github.com/user-attachments/assets/ce181b74-db29-4c1b-9beb-fd3212ac5cb2" />
<img width="1920" height="1080" alt="Screenshot (34)" src="https://github.com/user-attachments/assets/33b9bee9-b551-41a9-a7cf-a0fe55c40ba5" />
<img width="1920" height="1080" alt="Screenshot (33)" src="https://github.com/user-attachments/assets/8ffddd08-2eb8-48f8-a73e-0ffdcfe49f24" />


---

### Project Demo

# Video

🎥 Watch the demo here: [Demo Video](https://drive.google.com/file/d/1oGPGQho9Ab-yvJUPREHfqaSn5x5eOpH7/view?usp=drive_link)
This video shows real-time detection, classification, and shadow mimicry.


## Team Contributions

* Jerry Berna: Pose tracking, integration with Mediapipe
* Suha Shajahan: Trained the ML model using dataset
---
