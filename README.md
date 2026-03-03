💡 Hand Gesture Controlled Smart Light

Control your Tuya Smart Bulb using real-time hand gesture recognition with OpenCV and MediaPipe.

This project uses your webcam to detect the number of fingers shown and changes the bulb color accordingly.

-------------------------------------------------------------------------------------------------------

🚀 Features

✋ Real-time hand tracking using MediaPipe

🎨 Finger count based color control

💡 Smart bulb communication via TinyTuya

📷 Live webcam visualization with OpenCV

⚡ Sends command only when gesture changes (optimized communication)



🛠️ Technologies Used

Python

OpenCV

MediaPipe

TinyTuya

Computer Vision

IoT (Tuya Smart Bulb)



📌 How It Works

1) Webcam captures video frames.

2) MediaPipe detects hand landmarks.

3) Finger counting algorithm determines number of fingers raised.

4) Based on finger count, a color is selected.

5) TinyTuya sends RGB command to the smart bulb.

6) Color changes only when gesture changes (to prevent repeated commands).


🎨 Gesture → Color Mapping
Fingers Raised	                  Bulb Color
    1	                             🔴 Red
    2	                             🟡 Yellow
    3	                             🟢 Green
    4	                             🔵 Blue
    5	                             ⚪ White

    


Main Components
1️⃣ Tuya Bulb Setup

   Connects to smart bulb using:

          Device ID

          Device IP

          Local Key

   Uses TinyTuya library

2️⃣ Hand Detector Class

   Detects one hand

   Tracks landmarks

   Draws hand skeleton

3️⃣ Finger Counting Logic

  Thumb → Checked using X-axis comparison

  Other fingers → Checked using Y-axis comparison

  Returns total raised fingers

4️⃣ Color Control Function

  Maps finger count to RGB values and sends color command to bulb.

5️⃣ Main Loop

  Captures webcam frame

  Detects hand

  Counts fingers

  Changes bulb color

  Displays finger count on screen

  Press ESC to exit
