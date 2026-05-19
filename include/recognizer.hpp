#pragma once

// ============================================================
//  recognizer.hpp
//  Wraps OpenCV's LBPHFaceRecognizer with a simpler interface.
// ============================================================

#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// ── FaceRecognizer ────────────────────────────────────────────────────────────
//  Thin wrapper around cv::face::LBPHFaceRecognizer that handles the
//  train-vs-update distinction and exposes a clean predict interface.
//
//  LBPH algorithm summary:
//    1. Each pixel is compared to its 8 neighbours → 8-bit code (Local Binary Pattern).
//    2. LBP codes are histogrammed over a spatial grid of cells.
//    3. At prediction time the query histogram is compared to all stored
//       histograms via chi-squared distance (lower = more similar).
class FaceRecognizer {
public:
    FaceRecognizer();

    // Train the model on a batch of labelled grayscale face crops.
    // Call train() for the very first batch; update() for subsequent people
    // so the model grows incrementally without forgetting previous people.
    void train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);
    void update(const std::vector<cv::Mat>& images, const std::vector<int>& labels);

    // Predict the label for a single normalised face crop.
    // Returns {label, confidence}.  Confidence is a chi-squared distance —
    // lower means a better match.  Compare against CONFIDENCE_THRESH.
    struct Prediction { int label; double confidence; };
    Prediction predict(const cv::Mat& crop) const;

    bool isTrained() const { return trained_; }

    // Persist / restore the model so training survives between runs.
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    cv::Ptr<cv::face::LBPHFaceRecognizer> model_;
    bool trained_ = false;
};
