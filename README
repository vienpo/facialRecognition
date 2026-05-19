# FacialRec

Real-time face detection and recognition from a webcam, written in C++ using OpenCV's Haar cascades and LBPH recogniser.

![Demo — detect mode](docs/screenshots/detect.png)

---

## Features

- **Face detection** — draws a bounding box around every face in the frame
- **Landmark detection** — labels left eye, right eye, and mouth within each face
- **Live training** — press `T`, type a name, look at the camera; the model learns from 30 crops collected in real time
- **Recognition** — press `R` to identify trained people by name with a confidence score
- **Incremental learning** — train as many people as you like without restarting; the model grows in memory

---

## Demo

| Detect mode | Recognize mode |
|---|---|
| ![Detect](docs/screenshots/detect.png) | ![Recognize](docs/screenshots/recognize.png) |

---

## Architecture

```
facialRec/
├── include/
│   ├── config.hpp      # All tuneable constants (cascade path, thresholds, camera index)
│   ├── types.hpp       # Shared enums (Mode) and structs (AppState)
│   ├── detector.hpp    # Haar cascade loader + face/landmark detection
│   ├── recognizer.hpp  # LBPH recogniser wrapper
│   ├── renderer.hpp    # OpenCV drawing helpers (face box, landmark box, HUD)
│   └── utils.hpp       # Image preprocessing (prepareCrop)
├── src/
│   ├── main.cpp        # Entry point: webcam loop + keyboard input
│   ├── detector.cpp
│   ├── recognizer.cpp
│   ├── renderer.cpp
│   └── utils.cpp
├── CMakeLists.txt
├── requirements.txt
└── README.md
```

### How it works

1. **Detection** — `Detector::detectFaces()` runs `haarcascade_frontalface_default.xml` over a greyscale histogram-equalised frame using `detectMultiScale`.
2. **Landmarks** — `Detector::detectAndDrawLandmarks()` runs dedicated left/right eye cascades on the top 55% of each face ROI, and the smile cascade on the bottom 40%.  Restricting each cascade to its anatomical sub-region cuts false positives dramatically.
3. **Training** — `Utils::prepareCrop()` normalises each face to a 100×100 greyscale histogram-equalised image.  30 crops are collected live then fed to `cv::face::LBPHFaceRecognizer::train()` (first person) or `::update()` (subsequent people).
4. **Recognition** — `FaceRecognizer::predict()` returns a chi-squared distance.  Values below `CONFIDENCE_THRESH` (default 70) are treated as a match.

---

## Requirements

| Dependency | Version | Install |
|---|---|---|
| C++ compiler | C++17+ | `xcode-select --install` (macOS) / `sudo apt install build-essential` (Linux) |
| CMake | ≥ 3.16 | `brew install cmake` / `sudo apt install cmake` |
| OpenCV | ≥ 4.5 | `brew install opencv` / `sudo apt install libopencv-dev` |

See [`requirements.txt`](requirements.txt) for full details including building OpenCV from source.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/yourname/facialRec.git
cd facialRec

# 2. Set your cascade directory
#    Find it with:
find /opt/homebrew /usr -name "haarcascade_frontalface_default.xml" 2>/dev/null
#    Then edit CASCADE_DIR in include/config.hpp

# 3. Build
mkdir build && cd build
cmake ..
make -j$(nproc)        # Linux
make -j$(sysctl -n hw.logicalcpu)   # macOS

# 4. Run
./facialrec
```

---

## Controls

| Key | Action |
|---|---|
| `T` | Enter **Train** mode — you'll be prompted for a name in the terminal |
| `R` | Enter **Recognize** mode (requires at least one trained person) |
| `D` | Return to **Detect**-only mode |
| `Q` | Quit |

### Training a new person

1. Press `T` in the window
2. Type the person's name in the terminal and press Enter
3. Look at the camera — the green box counts up to 30 crops automatically
4. Training completes and the app returns to Detect mode
5. Press `R` to start recognising

---

## Configuration

All tuneable values are in [`include/config.hpp`](include/config.hpp):

| Constant | Default | Effect |
|---|---|---|
| `CASCADE_DIR` | *(your path)* | Folder containing Haar XML files |
| `SCALE_FACTOR` | `1.1` | Pyramid step — lower catches more faces but is slower |
| `MIN_NEIGHBORS` | `5` | Raise to reduce false positives |
| `MIN_FACE_SIZE` | `80` | Ignore faces smaller than this (px) |
| `TRAIN_SAMPLES` | `30` | Crops collected per person — more = better accuracy |
| `CONFIDENCE_THRESH` | `70.0` | LBPH distance cutoff — lower = stricter matching |
| `CAM_INDEX` | `0` | Webcam index (try 1, 2 … for external cameras) |

---

## Extending the project

**Better detection (DNN)**
Replace the Haar cascade with OpenCV's ResNet-based SSD face detector for more accurate detection of angled or partially occluded faces:
```cpp
cv::dnn::Net net = cv::dnn::readNetFromCaffe("deploy.prototxt", "res10.caffemodel");
```
Models: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

**Better recognition (deep embeddings)**
Replace LBPH with FaceNet/OpenFace embeddings compared by cosine similarity.  State-of-the-art accuracy on unconstrained faces.

**Persist the model**
Uncomment the two lines at the bottom of `main.cpp` to save/reload `model.xml` between runs.

**Multi-face tracking**
Add a `cv::TrackerCSRT` per detected face so identity labels stay stable between frames without re-running the classifier on every frame.

---

## License

MIT
