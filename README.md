<h1 align="center">
  🕹️ Hand Gesture Virtual Game Controller
</h1>

<h3 align="center">
  Control your game using your hands – with MediaPipe and a virtual gamepad!
</h3>

<p align="center">
  <img src="https://media.giphy.com/media/3o6ZtaO9BZHcOjmErm/giphy.gif" width="200">
</p>

<hr>

<h2>📷 Demo</h2>
<p><em>Insert a GIF or video of your system in action here.</em></p>

<hr>

<h2>🧠 Features</h2>

<ul>
  <li>Real-time hand tracking with <strong>MediaPipe</strong></li>
  <li>Gesture recognition using landmark angles and distances</li>
  <li>Virtual Xbox 360 controller emulation with <strong>vgamepad</strong></li>
  <li>Supports gestures like:
    <ul>
      <li>✊ Fist = Attack (Right hand)</li>
      <li>✌️ Index + Middle = Move Forward (Left hand)</li>
      <li>👍 + ☝️ Thumb + Index = Move Backward (Left hand)</li>
      <li>🖕 Index tracking = Camera control (Left hand)</li>
    </ul>
  </li>
</ul>

<hr>

<h2>🚀 Getting Started</h2>

<h3>🧰 Prerequisites</h3>

<ul>
  <li>Python 3.8+</li>
  <li>Webcam</li>
  <li>Xbox 360 Controller driver (for Windows)</li>
</ul>

<h3>📦 Install required packages</h3>

```bash
pip install opencv-python mediapipe vgamepad numpy
