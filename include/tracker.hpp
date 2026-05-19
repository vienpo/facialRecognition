#pragma once

// ============================================================
//  tracker.hpp
//  Multi-face tracker — assigns stable IDs to faces across frames.
// ============================================================
//
//  PROBLEM being solved:
//    Haar-cascade face detection runs independently every frame, so each
//    detected face is an anonymous bounding box — there's no continuity of
//    identity.  This means labels, emotion state, and recognition results
//    can flicker or swap between nearby faces.
//
//  SOLUTION — CSRT tracker per face:
//    On detection we pair each bounding box with a cv::TrackerCSRT instance
//    initialised on that region.  Every subsequent frame we call tracker->update()
//    which follows the face using correlation filters even without re-running the
//    face cascade.  We only re-run detection every TRACKER_REINIT_INTERVAL frames
//    to pick up new arrivals and fix any drift.
//
//  ID assignment (IoU matching):
//    When new detections arrive we match them to existing tracks by computing the
//    Intersection-over-Union (IoU) between the new bounding box and each tracked
//    box.  The highest-IoU pair above TRACKER_IOU_THRESH is a match.  Unmatched
//    detections become new tracks; tracks with no match accumulate framesLost.
//    Tracks that exceed TRACKER_MAX_LOST are discarded.

#include "types.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>

class FaceTracker {
public:
    // Update all existing tracks from the current frame.
    // Must be called every frame before reconcile().
    void update(const cv::Mat& frame);

    // Match a fresh set of face detections to existing tracks.
    // Creates new tracks for unmatched detections; marks others lost.
    // Call this every TRACKER_REINIT_INTERVAL frames (or on first frame).
    void reconcile(const cv::Mat& frame, const std::vector<cv::Rect>& detections);

    // Remove stale tracks (framesLost > TRACKER_MAX_LOST).
    void pruneLost();

    // Read-only access to current live tracks.
    const std::vector<TrackedFace>& tracks() const { return tracks_; }
    std::vector<TrackedFace>&       tracks()        { return tracks_; }

private:
    std::vector<TrackedFace> tracks_;
    int nextId_ = 0;

    // Compute Intersection-over-Union for two rectangles.
    static double iou(const cv::Rect& a, const cv::Rect& b);
};