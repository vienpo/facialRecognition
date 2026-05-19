// ============================================================
//  utils.cpp
//  Image preprocessing utilities.
// ============================================================

#include "utils.hpp"
#include "config.hpp"

namespace Utils {

// ── prepareCrop ───────────────────────────────────────────────────────────────
//  Produces a normalised face crop suitable for LBPH training and prediction.
//
//  Steps:
//    1. Clone the ROI (avoid modifying the original frame)
//    2. Convert to greyscale — LBPH only uses intensity information
//    3. Resize to fixed CROP_SIZE — LBPH histograms require consistent dimensions
//    4. Histogram equalisation — stretches contrast so the model is less sensitive
//       to changes in lighting between training and inference
//
//  This function must be called identically for training crops and prediction
//  crops.  Any mismatch in preprocessing will degrade recognition accuracy.
cv::Mat prepareCrop(const cv::Mat& frame, const cv::Rect& roi)
{
    cv::Mat crop = frame(roi).clone();
    cv::cvtColor(crop, crop, cv::COLOR_BGR2GRAY);
    cv::resize(crop, crop, {CROP_SIZE, CROP_SIZE});
    cv::equalizeHist(crop, crop);
    return crop;
}

} // namespace Utils
