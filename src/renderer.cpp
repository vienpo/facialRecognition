// ============================================================
//  renderer.cpp
//  OpenCV drawing helpers: face boxes, landmark boxes, HUD.
// ============================================================

#include "renderer.hpp"
#include "config.hpp"
#include <string>

namespace Renderer {

// ── drawFaceBox ───────────────────────────────────────────────────────────────
//  Draws a 2px rectangle around the face and a filled colour strip above it
//  so the label is always readable regardless of background colour.
void drawFaceBox(cv::Mat& frame, const cv::Rect& face,
                 const std::string& label, cv::Scalar color)
{
    cv::rectangle(frame, face, color, 2);

    // Measure text so the backing strip fits exactly
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                  0.6, 1, &baseline);
    cv::rectangle(frame,
                  {face.x,              face.y - ts.height - 8},
                  {face.x + ts.width + 4, face.y},
                  color, cv::FILLED);
    cv::putText(frame, label,
                {face.x + 2, face.y - 4},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 1);
}

// ── drawLandmarkBox ───────────────────────────────────────────────────────────
//  Thin 1px rectangle with a small label above it.  Kept lightweight so it
//  doesn't clutter the face box drawn by drawFaceBox.
void drawLandmarkBox(cv::Mat& frame, const cv::Rect& box,
                     const std::string& label, cv::Scalar color)
{
    cv::rectangle(frame, box, color, 1);
    cv::putText(frame, label,
                {box.x, box.y - 3},
                cv::FONT_HERSHEY_SIMPLEX, 0.38, color, 1);
}

// ── drawEmotionBadge ──────────────────────────────────────────────────────────
//  Renders below the face box:
//    • Emotion label (Happy / Neutral)
//    • A narrow horizontal bar showing the EMA smile score [0–1]
void drawEmotionBadge(cv::Mat& frame, const cv::Rect& face, const TrackedFace& tf)
{
    const int  barW    = face.width;
    const int  barH    = 6;
    const int  yBase   = face.y + face.height + 4;
    const int  yLabel  = yBase + barH + 12;

    // Background bar (dark grey)
    cv::rectangle(frame,
                  {face.x, yBase},
                  {face.x + barW, yBase + barH},
                  {50, 50, 50}, cv::FILLED);

    // Filled portion proportional to smile score
    int fillW = static_cast<int>(tf.smileScore * barW);
    cv::Scalar barColor = (tf.emotion == Emotion::HAPPY)
                          ? cv::Scalar{50, 220, 50}
                          : cv::Scalar{120, 120, 200};
    if (fillW > 0)
        cv::rectangle(frame,
                      {face.x, yBase},
                      {face.x + fillW, yBase + barH},
                      barColor, cv::FILLED);

    // Emotion label
    std::string label = (tf.emotion == Emotion::HAPPY) ? "Happy" : "Neutral";
    cv::putText(frame, label,
                {face.x, yLabel},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, barColor, 1);
}

// ── drawHUD ───────────────────────────────────────────────────────────────────
void drawHUD(cv::Mat& frame, const AppState& state)
{
    std::string modeStr;
    cv::Scalar  modeColor;

    switch (state.mode) {
        case Mode::DETECT:
            modeStr   = "MODE: DETECT";
            modeColor = {200, 200, 200};
            break;
        case Mode::TRAIN:
            modeStr   = "MODE: TRAIN  [" + state.trainName + "]  "
                        + std::to_string(state.trainCount) + "/"
                        + std::to_string(TRAIN_SAMPLES);
            modeColor = {50, 200, 50};
            break;
        case Mode::RECOGNIZE:
            modeStr   = "MODE: RECOGNIZE";
            modeColor = {50, 150, 255};
            break;
    }

    cv::putText(frame, modeStr, {10, 28},
                cv::FONT_HERSHEY_SIMPLEX, 0.75, modeColor, 2);
    cv::putText(frame, "T=Train  R=Recognise  D=Detect  S=Save model  Q=Quit",
                {10, frame.rows - 10},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, {180, 180, 180}, 1);
}

} // namespace Renderer
