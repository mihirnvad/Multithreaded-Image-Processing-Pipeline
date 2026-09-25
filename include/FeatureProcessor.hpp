#pragma once

#include "BoundedQueue.hpp"
#include "PosixThread.hpp"
#include "Types.hpp"

#include <opencv2/features2d.hpp>

#include <cstddef>
#include <vector>

namespace pipeline {

struct FeatureOptions {
    int max_features = 1000;    ///< ORB feature budget per image.
    bool feature_maps = false;  ///< Also render a keypoint-density heat map.
};

/// The per-thread feature-extraction kernel. Each worker owns its own
/// instance, so the ORB detector and scratch buffers are never shared between
/// threads and the hot path takes no locks at all.
class FeatureExtractor {
public:
    explicit FeatureExtractor(FeatureOptions options);

    /// Consumes the task (its pixel buffer is reused for the annotation) and
    /// returns the result: ORB keypoints + descriptors, the annotated image,
    /// image-quality statistics and timing.
    FrameResult process(FrameTask&& task, int worker_id);

private:
    FeatureOptions options_;
    cv::Ptr<cv::ORB> orb_;
    cv::Mat gray_;
    cv::Mat laplacian_;
    cv::Mat descriptors_;
    std::vector<cv::KeyPoint> keypoints_;
};

/// Stage 2: a pool of N worker threads. Each pops FrameTasks from the input
/// queue until it is closed and drained, and pushes FrameResults to the output
/// queue. The coordinator closes the output queue after joining the pool.
class FeatureProcessor {
public:
    FeatureProcessor(std::size_t num_workers, FeatureOptions options, BoundedQueue<FrameTask>& input,
                     BoundedQueue<FrameResult>& output);

    void start();
    void join();

    [[nodiscard]] std::size_t num_workers() const noexcept { return num_workers_; }
    [[nodiscard]] std::size_t images_processed() const noexcept;
    /// Images handled by each worker; shows how evenly work was distributed.
    [[nodiscard]] const std::vector<std::size_t>& per_worker_counts() const noexcept { return per_worker_counts_; }

private:
    void run(int worker_id);

    std::size_t num_workers_;
    FeatureOptions options_;
    BoundedQueue<FrameTask>& input_;
    BoundedQueue<FrameResult>& output_;
    std::vector<PosixThread> workers_;
    // Slot i is written only by worker i and read after join().
    std::vector<std::size_t> per_worker_counts_;
};

}  // namespace pipeline
