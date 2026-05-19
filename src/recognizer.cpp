// ============================================================
//  recognizer.cpp
//  LBPH face recogniser wrapper implementation.
// ============================================================

#include "recognizer.hpp"
#include <iostream>

FaceRecognizer::FaceRecognizer()
    : model_(cv::face::LBPHFaceRecognizer::create())
{}

// ── FaceRecognizer::train ────────────────────────────────────────────────────
//  Called for the very first person.  Resets any existing model state.
void FaceRecognizer::train(const std::vector<cv::Mat>& images,
                           const std::vector<int>& labels)
{
    model_->train(images, labels);
    trained_ = true;
}

// ── FaceRecognizer::update ───────────────────────────────────────────────────
//  Called for every subsequent person.  Appends to the existing model so
//  previously learned people are not forgotten (incremental learning).
void FaceRecognizer::update(const std::vector<cv::Mat>& images,
                            const std::vector<int>& labels)
{
    model_->update(images, labels);
}

// ── FaceRecognizer::predict ──────────────────────────────────────────────────
//  Returns the closest known label and its chi-squared distance.
//  The crop must have gone through Utils::prepareCrop() — same pipeline as
//  training — otherwise distances will be meaningless.
FaceRecognizer::Prediction FaceRecognizer::predict(const cv::Mat& crop) const
{
    int    label      = -1;
    double confidence = 0.0;
    model_->predict(crop, label, confidence);
    return {label, confidence};
}

// ── Persistence ──────────────────────────────────────────────────────────────
void FaceRecognizer::save(const std::string& path) const
{
    model_->save(path);
    std::cout << "[SAVE] Model written to " << path << "\n";
}

void FaceRecognizer::load(const std::string& path)
{
    model_->load(path);
    trained_ = true;
    std::cout << "[LOAD] Model loaded from " << path << "\n";
}
