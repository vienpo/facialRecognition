#pragma once

// ============================================================
//  detector.hpp
//  Loads Haar cascades and detects faces + facial landmarks.
// ============================================================

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// ── Detector ──────────────────────────────────────────────────────────────────
//  Wraps the four Haar cascade classifiers needed for face and landmark
//  detection.  Constructed once; reused every frame.
class Detector {
public:
    // Load all cascades from CASCADE_DIR defined in config.hpp.
    // Returns false and prints an error if any file is missing.
    bool load(const std::string& cascadeDir);

    // Detect all frontal faces in a greyscale frame.
    // Returns a vector of bounding boxes in frame coordinates.
    std::vector<cv::Rect> detectFaces(const cv::Mat& gray) const;

    // Detect left eye, right eye, and mouth within faceROI and draw coloured
    // boxes directly onto `frame`.  Searches sub-regions of the face to avoid
    // cross-face false positives (see detector.cpp for full explanation).
    void detectAndDrawLandmarks(cv::Mat& frame,
                                const cv::Mat& gray,
                                const cv::Rect& faceROI) const;

private:
    cv::CascadeClassifier faceCascade_;
    cv::CascadeClassifier eyeCascade_;
    cv::CascadeClassifier leftEyeCascade_;
    cv::CascadeClassifier rightEyeCascade_;
    cv::CascadeClassifier mouthCascade_;

    // Helper: load a single cascade and report success/failure.
    static bool loadOne(cv::CascadeClassifier& cc, const std::string& path);

    // Helper: detect within a sub-region and draw the best result.
    void detectInRegion(cv::Mat& frame,
                        const cv::Mat& gray,
                        const cv::Rect& region,
                        cv::CascadeClassifier& cascade,
                        const std::string& label,
                        cv::Scalar color,
                        cv::Size minSize,
                        int minNeighbors) const;
};
