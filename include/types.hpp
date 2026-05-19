#pragma once

// ============================================================
//  types.hpp
//  Shared enums and data structures used across modules.
// ============================================================

#include <map>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

// ── App mode ──────────────────────────────────────────────────────────────────
enum class Mode {
    DETECT,     // Draw a box around every face — no recognition
    TRAIN,      // Collect face crops to teach the recogniser a new person
    RECOGNIZE   // Identify faces against the trained model
};

// ── Emotion ───────────────────────────────────────────────────────────────────
//  Inferred from the smile cascade neighbour count smoothed over time.
enum class Emotion { NEUTRAL, HAPPY };

inline std::string emotionLabel(Emotion e) {
    return e == Emotion::HAPPY ? "Happy 😊" : "Neutral 😐";
}

// ── TrackedFace ───────────────────────────────────────────────────────────────
//  Represents one actively tracked face.  Created when a new face is first
//  detected; destroyed when the tracker goes stale for too long.
struct TrackedFace {
    int  id;                        // Unique track ID (monotonically increasing)
    cv::Ptr<cv::Tracker> tracker;   // CSRT tracker instance
    cv::Rect bbox;                  // Most recent bounding box (frame coordinates)

    // Recognition result (only meaningful in RECOGNIZE mode)
    int    label      = -1;
    double confidence = 0.0;

    // Emotion state
    Emotion emotion    = Emotion::NEUTRAL;
    double  smileScore = 0.0;       // EMA-smoothed smile strength [0, 1]

    int framesLost = 0;             // Consecutive frames the tracker has failed
};

// ── AppState ──────────────────────────────────────────────────────────────────
struct AppState {
    Mode mode = Mode::DETECT;

    // Training state
    int         nextLabel  = 0;
    int         trainCount = 0;
    std::string trainName;

    std::vector<cv::Mat> trainImages;
    std::vector<int>     trainLabels;

    std::map<int, std::string> labelToName;

    bool modelTrained = false;
};