// ============================================================
//  main.cpp
//  Entry point.  Owns the webcam, the main loop, and keyboard input.
//  All heavy logic is delegated to Detector, FaceRecognizer, and Renderer.
// ============================================================

#include <opencv2/opencv.hpp>
#include <iostream>

#include "config.hpp"
#include "types.hpp"
#include "detector.hpp"
#include "recognizer.hpp"
#include "renderer.hpp"
#include "utils.hpp"

int main()
{
    // ── Load cascades ─────────────────────────────────────────────────────────
    Detector detector;
    if (!detector.load(CASCADE_DIR)) {
        std::cerr << "[ERROR] Failed to load one or more cascades.\n"
                  << "        Edit CASCADE_DIR in include/config.hpp.\n";
        return 1;
    }

    // ── Initialise recogniser and app state ───────────────────────────────────
    FaceRecognizer recognizer;
    AppState state;

    // ── Open webcam ───────────────────────────────────────────────────────────
    cv::VideoCapture cap(CAM_INDEX);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open webcam (index " << CAM_INDEX << ").\n"
                  << "        Try changing CAM_INDEX in include/config.hpp.\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  CAM_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);
    std::cout << "[OK] Webcam opened at " << CAM_WIDTH << "×" << CAM_HEIGHT << ".\n"
              << "     T=Train  R=Recognise  D=Detect  Q=Quit\n";

    cv::Mat frame, gray;

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Greyscale + histogram equalisation used by all Haar cascades
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Detect all faces in the current frame
        auto faces = detector.detectFaces(gray);

        for (const auto& face : faces) {
            // Clamp to frame bounds (Haar can occasionally return a rect that
            // marginally exceeds the frame edge)
            cv::Rect safeROI = face & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safeROI.area() == 0) continue;

            cv::Mat crop = Utils::prepareCrop(frame, safeROI);

            // ── Per-mode face box ─────────────────────────────────────────────
            if (state.mode == Mode::TRAIN) {

                if (state.trainCount < TRAIN_SAMPLES) {
                    // Accumulate training crops
                    state.trainImages.push_back(crop);
                    state.trainLabels.push_back(state.nextLabel);
                    state.trainCount++;

                    Renderer::drawFaceBox(frame, safeROI,
                        "Collecting " + std::to_string(state.trainCount)
                        + "/" + std::to_string(TRAIN_SAMPLES),
                        {0, 255, 0});

                } else {
                    // Enough samples collected — commit to the model
                    if (recognizer.isTrained())
                        recognizer.update(state.trainImages, state.trainLabels);
                    else
                        recognizer.train(state.trainImages, state.trainLabels);

                    state.labelToName[state.nextLabel] = state.trainName;
                    std::cout << "[TRAIN] Label " << state.nextLabel
                              << " → \"" << state.trainName << "\" saved.\n";

                    // Reset training state and return to detect mode
                    state.nextLabel++;
                    state.trainImages.clear();
                    state.trainLabels.clear();
                    state.trainCount = 0;
                    state.mode = Mode::DETECT;
                }

            } else if (state.mode == Mode::RECOGNIZE && recognizer.isTrained()) {

                auto [label, confidence] = recognizer.predict(crop);
                bool known = (confidence < CONFIDENCE_THRESH);

                std::string name = known
                    ? state.labelToName.at(label)
                      + " (" + std::to_string(static_cast<int>(confidence)) + ")"
                    : "Unknown";

                Renderer::drawFaceBox(frame, safeROI, name,
                    known ? cv::Scalar{50,200,255} : cv::Scalar{0,0,220});

            } else {
                Renderer::drawFaceBox(frame, safeROI, "Face", {200, 200, 0});
            }

            // ── Landmark detection (runs in all modes) ────────────────────────
            detector.detectAndDrawLandmarks(frame, gray, safeROI);
        }

        // ── HUD overlay ───────────────────────────────────────────────────────
        Renderer::drawHUD(frame, state);
        cv::imshow("FacialRec", frame);

        // ── Keyboard input ────────────────────────────────────────────────────
        int key = cv::waitKey(1) & 0xFF;

        if (key == 'q' || key == 'Q') {
            break;

        } else if (key == 'd' || key == 'D') {
            state.mode = Mode::DETECT;
            std::cout << "[MODE] Detect\n";

        } else if (key == 'r' || key == 'R') {
            if (!recognizer.isTrained())
                std::cout << "[WARN] No model trained yet — press T first.\n";
            else {
                state.mode = Mode::RECOGNIZE;
                std::cout << "[MODE] Recognize\n";
            }

        } else if (key == 't' || key == 'T') {
            std::cout << "Enter name for new person: ";
            std::getline(std::cin, state.trainName);
            if (state.trainName.empty())
                state.trainName = "Person_" + std::to_string(state.nextLabel);
            state.trainCount = 0;
            state.trainImages.clear();
            state.trainLabels.clear();
            state.mode = Mode::TRAIN;
            std::cout << "[MODE] Training \"" << state.trainName
                      << "\" — look at the camera!\n";
        }
    }

    // ── Optional: persist model across runs ───────────────────────────────────
    //  Uncomment these two lines to save/reload the trained model:
    //
    //  if (recognizer.isTrained())
    //      recognizer.save("model.xml");

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
