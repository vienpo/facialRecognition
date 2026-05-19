#pragma once

// ============================================================
//  emotion.hpp
//  Heuristic emotion detection (Neutral / Happy).
// ============================================================
//
//  HOW IT WORKS:
//    The smile Haar cascade reports a "neighbour count" — how many overlapping
//    rectangle detections agreed on the smile region.  A higher count = stronger
//    smile signal.  We normalise this to [0, 1] and run it through an Exponential
//    Moving Average (EMA) to smooth out per-frame jitter.
//
//    If the smoothed score exceeds EMOTION_SMILE_THRESH (normalised), the face
//    is labelled Happy; otherwise Neutral.
//
//  LIMITATIONS:
//    - Binary only (Happy / Neutral).  Anger, sadness, surprise etc. require a
//      trained CNN — see docs/extending.md for pointers.
//    - Sensitive to lighting and head pose.  Works best front-on, good light.
//    - The threshold may need tuning per camera/lighting environment.

#include "types.hpp"
#include <opencv2/opencv.hpp>

namespace Emotion {

    // Analyse the mouth sub-region of a face and return a raw smile strength
    // in [0, 1].  0 = no smile detected; higher = stronger smile signal.
    //
    //  grayFrame  – full-frame greyscale image (histogram-equalised)
    //  faceROI    – bounding box of the face in frame coordinates
    //  mouthCasc  – pre-loaded smile cascade classifier
    double measureSmile(const cv::Mat& grayFrame,
                        const cv::Rect& faceROI,
                        cv::CascadeClassifier& mouthCasc);

    // Update the EMA smile score for a tracked face and classify the emotion.
    // Mutates tf.smileScore and tf.emotion in place.
    void update(TrackedFace& tf, double rawSmile);

} // namespace Emotion