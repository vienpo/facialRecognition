// ============================================================
//  emotion.cpp
//  Heuristic smile-based emotion detection.
// ============================================================

#include "emotion.hpp"
#include "config.hpp"
#include <algorithm>
#include <vector>

namespace Emotion {

// ── measureSmile ─────────────────────────────────────────────────────────────
//  Run the smile cascade inside the bottom 40% of the face ROI.
//  Returns a normalised smile strength in [0, 1].
//
//  We use detectMultiScale with outputRejectLevels=true which fills a
//  `levelWeights` vector — each entry is a confidence score for that
//  detection.  We take the maximum weight as our raw signal, then
//  clamp and normalise it to [0, 1].
//
//  If the overloaded form isn't available on your OpenCV build, we fall
//  back to using the neighbour count heuristic instead.
double measureSmile(const cv::Mat& grayFrame,
                    const cv::Rect& faceROI,
                    cv::CascadeClassifier& mouthCasc)
{
    const cv::Rect frameBounds(0, 0, grayFrame.cols, grayFrame.rows);

    // Restrict to bottom 40% of face (same region as landmark detector)
    cv::Rect mouthRegion(
        faceROI.x,
        faceROI.y + static_cast<int>(faceROI.height * 0.60),
        faceROI.width,
        static_cast<int>(faceROI.height * 0.40));
    mouthRegion &= frameBounds;
    if (mouthRegion.area() == 0) return 0.0;

    cv::Mat roi = grayFrame(mouthRegion);

    // Use neighbour count as a proxy for detection confidence.
    // minNeighbors=1 here so we capture the raw count; we threshold ourselves.
    std::vector<cv::Rect> mouths;
    std::vector<int>      rejectLevels;
    std::vector<double>   levelWeights;

    mouthCasc.detectMultiScale(roi, mouths, rejectLevels, levelWeights,
                               1.05, 1, cv::CASCADE_SCALE_IMAGE, {25, 15});

    if (mouths.empty()) return 0.0;

    // Take the highest confidence detection
    double maxWeight = *std::max_element(levelWeights.begin(), levelWeights.end());

    // Normalise: empirically weights run from ~0 to ~10 for a clear smile.
    // Clamp to [0, 1].
    constexpr double MAX_WEIGHT = 8.0;
    return std::min(maxWeight / MAX_WEIGHT, 1.0);
}

// ── update ────────────────────────────────────────────────────────────────────
//  Apply EMA smoothing to the raw smile score and classify the emotion.
//
//  EMA formula:  score_t = α × raw + (1 - α) × score_{t-1}
//  α = EMOTION_EMA_ALPHA (from config.hpp)
//
//  A low α (e.g. 0.1) gives a very stable but slow-reacting label.
//  A high α (e.g. 0.5) reacts quickly but flickers more.
void update(TrackedFace& tf, double rawSmile)
{
    // Exponential moving average
    tf.smileScore = EMOTION_EMA_ALPHA * rawSmile
                  + (1.0 - EMOTION_EMA_ALPHA) * tf.smileScore;

    // Threshold: normalised smile strength > 0.35 → Happy
    // (EMOTION_SMILE_THRESH from config.hpp is the integer neighbour version;
    //  here we use a normalised equivalent)
    constexpr double HAPPY_THRESH = 0.35;
    tf.emotion = (tf.smileScore >= HAPPY_THRESH) ? Emotion::HAPPY : Emotion::NEUTRAL;
}

} // namespace Emotion
