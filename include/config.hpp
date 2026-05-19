#pragma once

// ============================================================
//  config.hpp
//  Central configuration — edit this file to tune the app.
// ============================================================

#include <string>

// ── Cascade directory ─────────────────────────────────────────────────────────
//  Point this at the folder that contains the Haar XML files.
//
//  macOS Homebrew (find yours with):
//    find /opt/homebrew -name "haarcascade_frontalface_default.xml" 2>/dev/null
//
//  Linux:
//    /usr/share/opencv4/haarcascades/
//
inline const std::string CASCADE_DIR =
    "/opt/homebrew/Cellar/opencv/4.13.0_10/share/opencv4/haarcascades/";

// ── Detection parameters ──────────────────────────────────────────────────────
constexpr double SCALE_FACTOR  = 1.1;   // How aggressively the image is downsampled per
                                        // Haar pyramid level. Lower = slower but catches
                                        // more faces; higher = faster but may miss small ones.
constexpr int    MIN_NEIGHBORS = 5;     // How many adjacent detections must agree before a
                                        // face is confirmed. Raise to cut false positives.
constexpr int    MIN_FACE_SIZE = 80;    // Ignore faces smaller than this (pixels). Helps
                                        // ignore background false positives.

// ── Recognition parameters ───────────────────────────────────────────────────
constexpr int    CROP_SIZE         = 100;  // Normalised face crop size fed to LBPH (px).
constexpr int    TRAIN_SAMPLES     = 30;   // Number of face crops collected per person.
constexpr double CONFIDENCE_THRESH = 70.0; // LBPH chi-squared distance threshold.
                                           // Lower = stricter match. Tune per lighting.

// ── Camera settings ───────────────────────────────────────────────────────────
constexpr int CAM_INDEX  = 0;     // 0 = default webcam; try 1, 2 … for others
constexpr int CAM_WIDTH  = 1280;
constexpr int CAM_HEIGHT = 720;

// ── Model persistence ─────────────────────────────────────────────────────────
//  The LBPH model and label map are auto-saved here on quit (if trained) and
//  auto-loaded on startup (if the file exists).
inline const std::string MODEL_PATH      = "model.xml";
inline const std::string LABEL_MAP_PATH  = "labels.txt";

// ── Multi-face tracker ────────────────────────────────────────────────────────
//  CSRT (Channel and Spatial Reliability Tracking) is accurate and handles
//  partial occlusion well, at the cost of ~1–2 ms/face vs KCF.
//  Swap to cv::TrackerKCF::create() in tracker.cpp if you need lower latency.
constexpr int    TRACKER_REINIT_INTERVAL = 30;  // Re-run face detection every N frames
                                                 // to catch new arrivals and fix drift.
constexpr double TRACKER_IOU_THRESH      = 0.3; // Minimum IoU to match a detection to an
                                                 // existing track (0 = no overlap required,
                                                 // 1 = perfect overlap required).
constexpr int    TRACKER_MAX_LOST        = 45;  // Drop a track after this many consecutive
                                                 // frames without a matching detection.

// ── Emotion detection ─────────────────────────────────────────────────────────
//  Emotion is inferred from the mouth (smile) cascade neighbour count — a proxy
//  for smile strength — smoothed with an exponential moving average (EMA).
//  This is a heuristic, not a trained emotion classifier.
constexpr int    EMOTION_SMILE_THRESH  = 3;   // Minimum smile cascade neighbours to count
                                               // as "smiling".  Raise if too many false positives.
constexpr double EMOTION_EMA_ALPHA     = 0.2; // EMA smoothing factor (0–1).
                                               // Lower = smoother but slower to respond.
