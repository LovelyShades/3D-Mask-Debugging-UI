# 🥽 FaceMask_Lab (Web Demo)

[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Web%20Browser-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

A browser-based demo that overlays custom PNG masks on detected faces in **images** or a **live webcam feed**, using **MediaPipe Face Landmarker** + **TFJS FaceMesh** techniques.  
Includes a toggleable mesh wireframe and real-time blend shape scores for expressions.  

> **Author:** Alanna Matundan  
> **License:** Apache 2.0 — Portions © Google / MediaPipe  

---

## ✨ Features
- 🖼️ Detect face landmarks on uploaded images  
- 🎥 Real-time webcam face tracking  
- 🎭 Upload custom transparent PNG masks  
- 🕸️ Show/Hide landmark mesh overlay  
- 📊 Display live blend shape expression values  

---

## 🧰 Tech Stack
- **Language:** JavaScript (ES6+), HTML5, CSS3  
- **Libraries:** MediaPipe Tasks Vision (`@mediapipe/tasks-vision@0.10.3`)  
- **Rendering:** Canvas 2D API (with triangulated warping)  

---

## 🚀 Getting Started
### Prerequisites
- Modern browser with **WebGL** and **getUserMedia** support  
- Local HTTP server (due to model + asset loading)  

### Run Locally
```bash

# clone repo
git clone https://github.com/LovelyShades/FaceMask_Lab.git
cd FaceMask_Lab

# start local server (Python example)
python -m http.server 8000
Then open:
👉 http://localhost:8000

📖 Usage
Upload PNG mask → use test_mask.png or create your own (transparent).

Click on demo image (model.jpg) → detect + warp mask.

Enable Webcam → toggle predictions in real time.

Toggle Mesh → show or hide landmark wireframe.

View Blend Shapes → see expression intensity scores.

🖌️ Creating Custom Masks
Use mesh_map.jpg as the UV reference template

Align your design in Photoshop, GIMP, or Krita

Export as PNG with transparency

Upload via the file input

🧱 Project Structure

index.html       # main demo page
main.js          # detection + rendering logic
styles.css       # UI and layout styles
triangulation.js # landmark triangulation indices
uv_coords.js     # UV coordinates for mesh mapping
mesh_map.jpg     # template reference for custom masks
model.jpg        # sample image
test_mask.png    # test mask for demo

📚 What I Learned
Applying UV + triangulation mapping for texture warping

Using MediaPipe Face Landmarker with TFJS for browser apps

Integrating webcam APIs (getUserMedia) safely

Building modular, maintainable demo code with toggles and states

Designing user-friendly controls (upload, mesh toggle, live preview)

🛣️ Future Improvements
Multi-face support

More efficient rendering pipeline (WebGL / Three.js)

AR-style effects (stickers, filters, masks)

Save screenshots of masked faces

Blend shape → avatar animation demo

📜 License
Licensed under the Apache License 2.0.
Derived in part from MediaPipe Face Landmarker demos.
