// ============================================================
//  detector.cpp
//  Haar cascade loading and face/landmark detection logic.
// ============================================================

#include "detector.hpp"
#include "config.hpp"
#include "renderer.hpp"
#include <iostream>
#include <algorithm>

// ── Detector::load ────────────────────────────────────────────────────────────
bool Detector::load(const std::string& cascadeDir)
{
    return loadOne(faceCascade_,      cascadeDir + "haarcascade_frontalface_default.xml")
        && loadOne(eyeCascade_,       cascadeDir + "haarcascade_eye.xml")
        && loadOne(leftEyeCascade_,   cascadeDir + "haarcascade_lefteye_2splits.xml")
        && loadOne(rightEyeCascade_,  cascadeDir + "haarcascade_righteye_2splits.xml")
        && loadOne(mouthCascade_,     cascadeDir + "haarcascade_smile.xml");
}

bool Detector::loadOne(cv::CascadeClassifier& cc, const std::string& path)
{
    if (!cc.load(path)) {
        std::cerr << "[ERROR] Could not load cascade: " << path << "\n";
        return false;
    }
    std::cout << "[OK] Loaded: " << path << "\n";
    return true;
}

// ── Detector::detectFaces ────────────────────────────────────────────────────
std::vector<cv::Rect> Detector::detectFaces(const cv::Mat& gray) const
{
    // detectMultiScale slides a detection window at progressively smaller scales.
    //   scaleFactor  – pyramid downscale step (1.1 = 10% smaller each pass)
    //   minNeighbors – how many neighbouring rectangles must agree; higher = fewer
    //                  false positives but may miss smaller/tilted faces
    //   minSize      – ignore detections smaller than this
    std::vector<cv::Rect> faces;
    faceCascade_.detectMultiScale(gray, faces,
                                  SCALE_FACTOR, MIN_NEIGHBORS,
                                  cv::CASCADE_SCALE_IMAGE,
                                  {MIN_FACE_SIZE, MIN_FACE_SIZE});
    return faces;
}

// ── Detector::detectAndDrawLandmarks ─────────────────────────────────────────
//
//  WHY search inside the face ROI?
//    Running the cascade on the full frame would fire on eyes/mouths in the
//    background or on other people's faces.  Constraining the search to the
//    known face bounding box is both faster and far more precise.
//
//  WHY split into upper/lower sub-regions?
//    Eyes live in the top ~55% of the face; the mouth in the bottom ~40%.
//    If we search the whole face for eyes, the cascade can misfire on mouth
//    corners and chin creases.  Restricting each cascade to its anatomical
//    region eliminates most false positives at zero extra cost.
//
void Detector::detectAndDrawLandmarks(cv::Mat& frame,
                                      const cv::Mat& gray,
                                      const cv::Rect& faceROI) const
{
    const cv::Rect frameBounds(0, 0, frame.cols, frame.rows);

    // ── Eyes: top 55% of face, split left/right ───────────────────────────────
    //  We use the dedicated left/right-eye cascades rather than the generic eye
    //  cascade so each detection gets a correct "left eye" / "right eye" label.
    //  Note: "left" and "right" are from the viewer's perspective.
    cv::Rect eyeStrip(faceROI.x, faceROI.y,
                      faceROI.width,
                      static_cast<int>(faceROI.height * 0.55));
    eyeStrip &= frameBounds;

    if (eyeStrip.area() > 0) {
        cv::Rect leftHalf(eyeStrip.x, eyeStrip.y,
                          eyeStrip.width / 2, eyeStrip.height);
        cv::Rect rightHalf(eyeStrip.x + eyeStrip.width / 2, eyeStrip.y,
                           eyeStrip.width / 2, eyeStrip.height);

        detectInRegion(frame, gray, leftHalf  & frameBounds,
                       leftEyeCascade_,  "left eye",  {255, 220, 50}, {18,18}, 6);
        detectInRegion(frame, gray, rightHalf & frameBounds,
                       rightEyeCascade_, "right eye", {255, 220, 50}, {18,18}, 6);
    }

    // ── Mouth: bottom 40% of face ─────────────────────────────────────────────
    //  minNeighbors=18 is intentionally high — haarcascade_smile.xml fires
    //  aggressively on chin/neck texture, so we demand strong consensus.
    cv::Rect mouthRegion(faceROI.x,
                         faceROI.y + static_cast<int>(faceROI.height * 0.60),
                         faceROI.width,
                         static_cast<int>(faceROI.height * 0.40));
    mouthRegion &= frameBounds;

    if (mouthRegion.area() > 0)
        detectInRegion(frame, gray, mouthRegion,
                       mouthCascade_, "mouth", {80, 120, 255}, {25,15}, 18);
}

// ── Detector::detectInRegion ─────────────────────────────────────────────────
//  Runs a cascade on a sub-region of the frame and draws only the single
//  best (largest) result.  Coordinates are translated back to frame space.
void Detector::detectInRegion(cv::Mat& frame,
                               const cv::Mat& gray,
                               const cv::Rect& region,
                               cv::CascadeClassifier& cascade,
                               const std::string& label,
                               cv::Scalar color,
                               cv::Size minSize,
                               int minNeighbors) const
{
    if (region.area() == 0) return;

    cv::Mat roi = gray(region);
    std::vector<cv::Rect> found;
    cascade.detectMultiScale(roi, found, 1.1, minNeighbors,
                             cv::CASCADE_SCALE_IMAGE, minSize);
    if (found.empty()) return;

    // Pick the largest detection (most confident)
    auto& best = *std::max_element(found.begin(), found.end(),
        [](const cv::Rect& a, const cv::Rect& b){ return a.area() < b.area(); });

    // Translate from ROI-local coordinates back to full-frame coordinates
    cv::Rect absBox(region.x + best.x, region.y + best.y, best.width, best.height);
    Renderer::drawLandmarkBox(frame, absBox, label, color);
}
