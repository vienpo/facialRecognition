#pragma once

// ============================================================
//  renderer.hpp
//  All OpenCV drawing helpers: face boxes, landmark boxes, HUD.
// ============================================================

#include <opencv2/opencv.hpp>
#include <string>
#include "types.hpp"

namespace Renderer {

    // Draw a thick labelled rectangle around a detected face.
    void drawFaceBox(cv::Mat& frame,
                     const cv::Rect& face,
                     const std::string& label,
                     cv::Scalar color);

    // Draw a thin labelled rectangle for a facial landmark (eye / mouth).
    void drawLandmarkBox(cv::Mat& frame,
                         const cv::Rect& box,
                         const std::string& label,
                         cv::Scalar color);

    // Draw the emotion badge (Happy / Neutral) below the face box.
    // Also renders a small smile-score bar so you can see the raw signal.
    void drawEmotionBadge(cv::Mat& frame,
                          const cv::Rect& face,
                          const TrackedFace& tf);

    // Render the heads-up display.
    void drawHUD(cv::Mat& frame, const AppState& state);

} // namespace Renderer