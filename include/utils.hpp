#pragma once

// ============================================================
//  utils.hpp
//  Image preprocessing utilities shared across modules.
// ============================================================

#include <opencv2/opencv.hpp>

namespace Utils {

    // Normalise a face crop for use as LBPH training/prediction input:
    //   1. Convert to greyscale  (LBPH is intensity-only)
    //   2. Resize to CROP_SIZE × CROP_SIZE  (fixed input dimensions)
    //   3. Histogram-equalise   (reduces sensitivity to lighting changes)
    //
    // Consistent preprocessing is critical — the crop fed to predict()
    // must go through the exact same pipeline as the training crops.
    cv::Mat prepareCrop(const cv::Mat& frame, const cv::Rect& roi);

} // namespace Utils
