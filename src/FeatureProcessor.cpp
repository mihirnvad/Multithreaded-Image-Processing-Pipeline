#include "FeatureProcessor.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace pipeline {

namespace {

const cv::Scalar kKeypointColor(0, 255, 0);  // BGR green

/// Renders a heat map of keypoint density at quarter resolution: each keypoint
/// deposits unit mass, a Gaussian spreads it, and a perceptual colormap makes
/// feature-rich and feature-poor regions obvious at a glance.
cv::Mat render_feature_map(const std::vector<cv::KeyPoint>& keypoints, cv::Size image_size) {
    constexpr int kScale = 4;
    const int rows = std::max(1, image_size.height / kScale);
    const int cols = std::max(1, image_size.width / kScale);
    // Constructed directly rather than via cv::Mat::zeros(): the MatExpr path
    // goes through a lazily-initialized singleton inside libopencv_core.
    cv::Mat density(rows, cols, CV_32F, cv::Scalar(0));
    for (const cv::KeyPoint& kp : keypoints) {
        const int r = std::clamp(static_cast<int>(kp.pt.y) / kScale, 0, rows - 1);
        const int c = std::clamp(static_cast<int>(kp.pt.x) / kScale, 0, cols - 1);
        density.at<float>(r, c) += 1.0f;
    }
    const double sigma = std::max(rows, cols) / 40.0;
    cv::GaussianBlur(density, density, cv::Size(0, 0), sigma);
    cv::Mat density_u8;
    cv::normalize(density, density_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat heat;
    cv::applyColorMap(density_u8, heat, cv::COLORMAP_INFERNO);
    return heat;
}

}  // namespace

FeatureExtractor::FeatureExtractor(FeatureOptions options)
    : options_(options), orb_(cv::ORB::create(options.max_features)) {}

FrameResult FeatureExtractor::process(FrameTask&& task, int worker_id) {
    const TimePoint start = Clock::now();

    FrameResult result;
    result.sequence = task.sequence;
    result.filename = std::move(task.filename);
    result.worker_id = worker_id;
    result.width = task.raw_image.cols;
    result.height = task.raw_image.rows;
    result.read_start = task.read_start;
    result.read_time_ms = task.read_time_ms;
    result.queue_wait_ms = elapsed_ms(task.read_timestamp, start);

    // Feature extraction: ORB keypoints + 256-bit binary descriptors.
    cv::cvtColor(task.raw_image, gray_, cv::COLOR_BGR2GRAY);
    keypoints_.clear();
    orb_->detectAndCompute(gray_, cv::noArray(), keypoints_, descriptors_);
    result.num_keypoints = static_cast<int>(keypoints_.size());

    // Image-quality statistics used by the visualization module.
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(gray_, mean, stddev);
    result.brightness = mean[0];
    result.contrast = stddev[0];
    cv::Laplacian(gray_, laplacian_, CV_32F);
    cv::meanStdDev(laplacian_, mean, stddev);
    result.sharpness = stddev[0] * stddev[0];

    if (options_.feature_maps) {
        result.feature_map = render_feature_map(keypoints_, task.raw_image.size());
    }

    // Draw keypoints in place over the raw pixels. The task owns the only
    // reference to this buffer, so annotating it avoids allocating and
    // copying a second full-resolution image.
    cv::drawKeypoints(task.raw_image, keypoints_, task.raw_image, kKeypointColor,
                      cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    result.annotated_image = std::move(task.raw_image);

    result.processed_timestamp = Clock::now();
    result.process_time_ms = elapsed_ms(start, result.processed_timestamp);
    return result;
}

FeatureProcessor::FeatureProcessor(std::size_t num_workers, FeatureOptions options, BoundedQueue<FrameTask>& input,
                                   BoundedQueue<FrameResult>& output)
    : num_workers_(num_workers),
      options_(options),
      input_(input),
      output_(output),
      per_worker_counts_(num_workers, 0) {
    if (num_workers_ == 0) {
        throw std::invalid_argument("FeatureProcessor needs at least one worker");
    }
}

void FeatureProcessor::start() {
    workers_.reserve(num_workers_);
    for (std::size_t i = 0; i < num_workers_; ++i) {
        const int id = static_cast<int>(i);
        workers_.emplace_back("worker-" + std::to_string(id), [this, id] { run(id); });
    }
}

void FeatureProcessor::join() {
    for (PosixThread& worker : workers_) {
        worker.join();
    }
}

std::size_t FeatureProcessor::images_processed() const noexcept {
    return std::accumulate(per_worker_counts_.begin(), per_worker_counts_.end(), std::size_t{0});
}

void FeatureProcessor::run(int worker_id) {
    FeatureExtractor extractor(options_);
    std::size_t& processed = per_worker_counts_[static_cast<std::size_t>(worker_id)];
    // pop() returns std::nullopt only once the reader has closed the queue and
    // every buffered task has been taken, so no work is ever dropped.
    while (std::optional<FrameTask> task = input_.pop()) {
        FrameResult result = extractor.process(std::move(*task), worker_id);
        if (!output_.push(std::move(result))) {
            break;  // writer side shut down
        }
        ++processed;
    }
}

}  // namespace pipeline
