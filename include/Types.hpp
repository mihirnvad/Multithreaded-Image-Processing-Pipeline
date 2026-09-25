#pragma once

#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <string>

namespace pipeline {

// steady_clock is monotonic; high_resolution_clock is an alias for
// system_clock on libstdc++ and can jump when NTP adjusts the wall clock,
// which would corrupt interval measurements.
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

inline double elapsed_ms(TimePoint from, TimePoint to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

/// Unit of work flowing from ImageReader to FeatureProcessor.
/// Moved through the queue: cv::Mat is a ref-counted header, so moving it
/// transfers ownership of the pixel buffer without copying a single byte.
struct FrameTask {
    std::uint64_t sequence = 0;  ///< Position in the sorted input listing.
    std::string filename;        ///< File name relative to the input directory.
    cv::Mat raw_image;           ///< Decoded BGR image.
    TimePoint read_start{};      ///< Just before cv::imread; origin of end-to-end latency.
    TimePoint read_timestamp{};  ///< Just after decoding, when the task entered the queue.
    double read_time_ms = 0.0;   ///< Disk read + JPEG/PNG decode time.
};

/// Unit of work flowing from FeatureProcessor to ImageWriter.
struct FrameResult {
    std::uint64_t sequence = 0;
    std::string filename;
    cv::Mat annotated_image;  ///< Input image with ORB keypoints drawn in green.
    cv::Mat feature_map;      ///< Keypoint-density heat map (empty unless requested).
    int num_keypoints = 0;
    int worker_id = -1;  ///< Worker that processed the frame, -1 in sequential mode.
    int width = 0;
    int height = 0;

    // Image-quality statistics, computed on the grayscale image.
    double sharpness = 0.0;   ///< Variance of the Laplacian; low values mean blur.
    double brightness = 0.0;  ///< Mean intensity, 0-255.
    double contrast = 0.0;    ///< Standard deviation of intensity.

    // Per-stage timings in milliseconds.
    TimePoint read_start{};
    TimePoint processed_timestamp{};  ///< When the result entered the output queue.
    double read_time_ms = 0.0;
    double queue_wait_ms = 0.0;  ///< Time spent in the input queue waiting for a worker.
    double process_time_ms = 0.0;
    double write_time_ms = 0.0;
    double total_latency_ms = 0.0;  ///< From read start until the result is written.
};

}  // namespace pipeline
