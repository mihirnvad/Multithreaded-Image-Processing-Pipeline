#pragma once

#include "PosixThread.hpp"

#include <signal.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace pipeline {

struct PipelineConfig {
    std::filesystem::path input_dir;
    std::optional<std::filesystem::path> output_dir;
    std::filesystem::path metrics_path = "metrics.csv";
    std::size_t num_workers = 1;
    std::size_t queue_capacity = 32;
    int max_features = 1000;
    bool feature_maps = false;  ///< Render keypoint heat maps (needs output_dir).
    bool sequential = false;    ///< Single-threaded baseline: no queues, no worker threads.
    std::size_t limit = 0;      ///< Process at most this many images; 0 means all.
};

struct PipelineStats {
    std::string mode;
    std::size_t workers = 0;
    std::size_t images_found = 0;
    std::size_t images_processed = 0;
    std::size_t read_failures = 0;
    std::size_t write_failures = 0;
    bool interrupted = false;
    double wall_time_ms = 0.0;
    double throughput_ips = 0.0;  ///< Images per second over the wall time.
    double mean_process_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    std::vector<std::size_t> per_worker_counts;

    [[nodiscard]] std::string to_json() const;
};

/// Coordinator. Wires reader -> [input queue] -> N workers -> [output queue]
/// -> writer, starts every thread, and tears them down in dependency order:
///
///   1. the reader finishes (or sees a stop request) and closes the input queue;
///   2. workers drain the input queue, then exit; the coordinator joins them
///      and closes the output queue;
///   3. the writer drains the output queue, flushes the CSV and exits.
///
/// Each queue is closed by exactly one party once all of its producers are
/// done, so shutdown cannot deadlock and no in-flight frame is lost.
class Pipeline {
public:
    explicit Pipeline(PipelineConfig config);

    /// Runs to completion (or until request_stop) and returns the statistics.
    PipelineStats run();

    /// Thread-safe. The reader stops at the next image boundary; frames that
    /// are already queued are still processed and written.
    void request_stop() noexcept { stop_requested_.store(true, std::memory_order_relaxed); }

    [[nodiscard]] bool stop_requested() const noexcept { return stop_requested_.load(std::memory_order_relaxed); }

private:
    PipelineStats run_parallel(const std::vector<std::filesystem::path>& files);
    PipelineStats run_sequential(const std::vector<std::filesystem::path>& files);

    PipelineConfig config_;
    std::atomic<bool> stop_requested_{false};
};

/// Turns SIGINT / SIGTERM into a callback on an ordinary thread.
///
/// Construct it before creating any other thread: it blocks those signals in
/// the calling thread, every thread created afterwards inherits that mask, and
/// a dedicated thread receives them synchronously with sigwait(). The callback
/// therefore runs in normal thread context rather than inside an
/// async-signal handler, so it may take locks or log. A second signal
/// terminates the process immediately with exit code 130.
class ScopedSignalHandler {
public:
    explicit ScopedSignalHandler(std::function<void(int)> on_signal);
    ~ScopedSignalHandler();

    ScopedSignalHandler(const ScopedSignalHandler&) = delete;
    ScopedSignalHandler& operator=(const ScopedSignalHandler&) = delete;

private:
    void run();

    std::function<void(int)> on_signal_;
    sigset_t signals_{};
    sigset_t previous_mask_{};
    std::atomic<bool> shutting_down_{false};
    PosixThread thread_;
};

}  // namespace pipeline
