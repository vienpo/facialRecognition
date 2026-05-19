// ============================================================
//  tracker.cpp
//  Multi-face CSRT tracker implementation.
// ============================================================

#include "tracker.hpp"
#include "config.hpp"
#include <algorithm>
#include <limits>

// ── FaceTracker::update ───────────────────────────────────────────────────────
//  Advance every live tracker by one frame.  If the tracker fails (returns
//  false) we increment framesLost so the track can be pruned later.
void FaceTracker::update(const cv::Mat& frame)
{
    for (auto& tf : tracks_) {
        cv::Rect box;
        bool ok = tf.tracker->update(frame, box);
        if (ok) {
            // Clamp to frame bounds to avoid out-of-range crop later
            tf.bbox = box & cv::Rect(0, 0, frame.cols, frame.rows);
            tf.framesLost = 0;
        } else {
            tf.framesLost++;
        }
    }
}

// ── FaceTracker::reconcile ────────────────────────────────────────────────────
//  Match fresh face detections to existing tracks using greedy IoU matching.
//
//  Algorithm:
//    1. Build a cost matrix of IoU scores (detections × tracks).
//    2. Greedily pick the highest IoU pair above the threshold and mark both
//       as matched.
//    3. Unmatched detections → new tracks.
//    4. Unmatched tracks → framesLost++ (they may recover next cycle).
void FaceTracker::reconcile(const cv::Mat& frame,
                            const std::vector<cv::Rect>& detections)
{
    if (detections.empty()) return;

    std::vector<bool> detMatched(detections.size(), false);
    std::vector<bool> trkMatched(tracks_.size(),    false);

    // Greedy IoU matching
    // For each detection find the best unmatched track
    for (size_t di = 0; di < detections.size(); ++di) {
        double bestIou  = TRACKER_IOU_THRESH;
        int    bestTrk  = -1;

        for (size_t ti = 0; ti < tracks_.size(); ++ti) {
            if (trkMatched[ti]) continue;
            double score = iou(detections[di], tracks_[ti].bbox);
            if (score > bestIou) {
                bestIou = score;
                bestTrk = static_cast<int>(ti);
            }
        }

        if (bestTrk >= 0) {
            // Re-initialise the tracker on the fresh detection bbox — this
            // corrects any drift that accumulated since the last reconcile.
            tracks_[bestTrk].bbox = detections[di];
            tracks_[bestTrk].tracker = cv::TrackerCSRT::create();
            tracks_[bestTrk].tracker->init(frame, detections[di]);
            tracks_[bestTrk].framesLost = 0;
            trkMatched[bestTrk] = true;
            detMatched[di]      = true;
        }
    }

    // Unmatched tracks: increment loss counter
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        if (!trkMatched[ti])
            tracks_[ti].framesLost++;
    }

    // Unmatched detections: spawn new tracks
    for (size_t di = 0; di < detections.size(); ++di) {
        if (detMatched[di]) continue;

        TrackedFace tf;
        tf.id      = nextId_++;
        tf.bbox    = detections[di];
        tf.tracker = cv::TrackerCSRT::create();
        tf.tracker->init(frame, detections[di]);
        tracks_.push_back(std::move(tf));
    }
}

// ── FaceTracker::pruneLost ────────────────────────────────────────────────────
void FaceTracker::pruneLost()
{
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const TrackedFace& tf){
                           return tf.framesLost > TRACKER_MAX_LOST;
                       }),
        tracks_.end());
}

// ── FaceTracker::iou ──────────────────────────────────────────────────────────
//  IoU = area(intersection) / area(union)
//  Returns 0 if the rectangles don't overlap.
double FaceTracker::iou(const cv::Rect& a, const cv::Rect& b)
{
    cv::Rect inter = a & b;
    if (inter.area() <= 0) return 0.0;
    double unionArea = static_cast<double>(a.area() + b.area() - inter.area());
    return unionArea > 0 ? inter.area() / unionArea : 0.0;
}
