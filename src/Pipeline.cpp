#include "Pipeline.hpp"

#include "BoundedQueue.hpp"
#include "FeatureProcessor.hpp"
#include "ImageReader.hpp"
#include "ImageWriter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace pipeline {

namespace fs = std::filesystem;

namespace {

/// Nearest-rank percentile; `values` is taken by copy because it is sorted.
double percentile(std::vector<double> values, double pct) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank = std::ceil(pct / 100.0 * static_cast<double>(values.size()));
    const std::size_t index = static_cast<std::size_t>(std::max(1.0, rank)) - 1;
    return values[std::min(index, values.size() - 1)];
}

void fill_writer_stats(PipelineStats& stats, const ImageWriter& writer, double wall_ms) {
    stats.images_processed = writer.images_written();
    stats.write_failures = writer.write_failures();
    stats.wall_time_ms = wall_ms;
    stats.throughput_ips = wall_ms > 0.0 ? 1000.0 * static_cast<double>(stats.images_processed) / wall_ms : 0.0;
    const auto& process = writer.process_times_ms();
    stats.mean_process_ms =
        process.empty() ? 0.0 : std::accumulate(process.begin(), process.end(), 0.0) / static_cast<double>(process.size());
    stats.p50_latency_ms = percentile(writer.total_latencies_ms(), 50);
    stats.p95_latency_ms = percentile(writer.total_latencies_ms(), 95);
    stats.p99_latency_ms = percentile(writer.total_latencies_ms(), 99);
}

}  // namespace

std::string PipelineStats::to_json() const {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    out << "{\"mode\":\"" << mode << "\",\"workers\":" << workers << ",\"images_found\":" << images_found
        << ",\"images_processed\":" << images_processed << ",\"read_failures\":" << read_failures
        << ",\"write_failures\":" << write_failures << ",\"interrupted\":" << (interrupted ? "true" : "false")
        << ",\"wall_time_ms\":" << wall_time_ms << ",\"throughput_ips\":" << throughput_ips
        << ",\"mean_process_ms\":" << mean_process_ms << ",\"p50_latency_ms\":" << p50_latency_ms
        << ",\"p95_latency_ms\":" << p95_latency_ms << ",\"p99_latency_ms\":" << p99_latency_ms
        << ",\"per_worker_counts\":[";
    for (std::size_t i = 0; i < per_worker_counts.size(); ++i) {
        out << (i ? "," : "") << per_worker_counts[i];
    }
    out << "]}";
    return out.str();
}

Pipeline::Pipeline(PipelineConfig config) : config_(std::move(config)) {
    if (!fs::is_directory(config_.input_dir)) {
        throw std::invalid_argument("input directory does not exist: " + config_.input_dir.string());
    }
    if (config_.num_workers == 0) {
        throw std::invalid_argument("need at least one worker thread");
    }
    if (config_.queue_capacity == 0) {
        throw std::invalid_argument("queue capacity must be > 0");
    }
}

PipelineStats Pipeline::run() {
    std::vector<fs::path> files = ImageReader::list_images(config_.input_dir);
    if (config_.limit > 0 && files.size() > config_.limit) {
        files.resize(config_.limit);
    }
    PipelineStats stats = config_.sequential ? run_sequential(files) : run_parallel(files);
    stats.images_found = files.size();
    stats.interrupted = stop_requested();
    return stats;
}

PipelineStats Pipeline::run_parallel(const std::vector<fs::path>& files) {
    BoundedQueue<FrameTask> input_queue(config_.queue_capacity);
    BoundedQueue<FrameResult> output_queue(config_.queue_capacity);

    WriterOptions writer_options{config_.output_dir, config_.metrics_path, 90};
    FeatureOptions feature_options{config_.max_features, config_.feature_maps && config_.output_dir.has_value()};

    ImageReader reader(config_.input_dir, files, input_queue, stop_requested_);
    FeatureProcessor processor(config_.num_workers, feature_options, input_queue, output_queue);
    ImageWriter writer(std::move(writer_options), output_queue);

    const TimePoint start = Clock::now();
    // Start consumers before producers so nothing waits on a thread that
    // does not exist yet.
    writer.start();
    processor.start();
    reader.start();

    reader.join();        // closes input_queue on exit
    processor.join();     // all workers saw "closed and drained"
    output_queue.close(); // last producer is gone: end of stream for the writer
    writer.join();        // flushed metrics.csv
    const double wall_ms = elapsed_ms(start, Clock::now());

    PipelineStats stats;
    stats.mode = "parallel";
    stats.workers = processor.num_workers();
    stats.read_failures = reader.failures();
    stats.per_worker_counts = processor.per_worker_counts();
    fill_writer_stats(stats, writer, wall_ms);
    return stats;
}

PipelineStats Pipeline::run_sequential(const std::vector<fs::path>& files) {
    // Same kernels as the parallel path, called inline on one thread with no
    // queues or locks: the honest baseline for measuring speedup.
    BoundedQueue<FrameResult> unused_queue(1);
    WriterOptions writer_options{config_.output_dir, config_.metrics_path, 90};
    ImageWriter writer(std::move(writer_options), unused_queue);
    FeatureExtractor extractor(
        FeatureOptions{config_.max_features, config_.feature_maps && config_.output_dir.has_value()});

    PipelineStats stats;
    stats.mode = "sequential";
    stats.workers = 1;

    const TimePoint start = Clock::now();
    std::uint64_t sequence = 0;
    for (const fs::path& path : files) {
        if (stop_requested()) {
            break;
        }
        std::optional<FrameTask> task =
            ImageReader::load(path, path.lexically_relative(config_.input_dir).generic_string(), sequence++);
        if (!task) {
            ++stats.read_failures;
            std::fprintf(stderr, "warning: could not decode %s, skipping\n", path.string().c_str());
            continue;
        }
        FrameResult result = extractor.process(std::move(*task), -1);
        writer.write(result);
    }
    writer.finish();
    const double wall_ms = elapsed_ms(start, Clock::now());

    stats.per_worker_counts = {writer.images_written()};
    fill_writer_stats(stats, writer, wall_ms);
    return stats;
}

// ---------------------------------------------------------------------------
// ScopedSignalHandler
// ---------------------------------------------------------------------------

ScopedSignalHandler::ScopedSignalHandler(std::function<void(int)> on_signal) : on_signal_(std::move(on_signal)) {
    sigemptyset(&signals_);
    sigaddset(&signals_, SIGINT);
    sigaddset(&signals_, SIGTERM);
    sigaddset(&signals_, SIGUSR1);  // private wake-up used by the destructor
    const int rc = pthread_sigmask(SIG_BLOCK, &signals_, &previous_mask_);
    if (rc != 0) {
        throw std::system_error(rc, std::generic_category(), "pthread_sigmask");
    }
    thread_.start("signals", [this] { run(); });
}

ScopedSignalHandler::~ScopedSignalHandler() {
    shutting_down_.store(true, std::memory_order_release);
    pthread_kill(thread_.native_handle(), SIGUSR1);
    thread_.join();
    pthread_sigmask(SIG_SETMASK, &previous_mask_, nullptr);
}

void ScopedSignalHandler::run() {
    bool stop_already_requested = false;
    for (;;) {
        int signo = 0;
        if (sigwait(&signals_, &signo) != 0) {
            continue;
        }
        if (signo == SIGUSR1) {
            if (shutting_down_.load(std::memory_order_acquire)) {
                return;
            }
            continue;  // stray SIGUSR1 from outside: ignore
        }
        if (stop_already_requested) {
            static constexpr char kMsg[] = "\nsecond interrupt, exiting immediately\n";
            const ssize_t ignored = ::write(STDERR_FILENO, kMsg, sizeof(kMsg) - 1);
            (void)ignored;
            _exit(130);
        }
        stop_already_requested = true;
        on_signal_(signo);
    }
}

}  // namespace pipeline
