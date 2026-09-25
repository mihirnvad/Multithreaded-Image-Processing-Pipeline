#pragma once

#include "BoundedQueue.hpp"
#include "PosixThread.hpp"
#include "Types.hpp"

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace pipeline {

struct WriterOptions {
    std::optional<std::filesystem::path> output_dir;  ///< Save annotated images here if set.
    std::filesystem::path metrics_path = "metrics.csv";
    int jpeg_quality = 90;
};

/// Stage 3: a single consumer thread that drains the output queue, optionally
/// saves annotated images and feature maps, and appends one CSV row of
/// microsecond-resolution timing and quality metrics per image.
///
/// Because only this thread touches the CSV file and the output directory, no
/// locking is needed around I/O: the queue is the only synchronization point.
class ImageWriter {
public:
    /// Opens the metrics file and creates the output directory; throws
    /// std::runtime_error on failure so misconfiguration fails fast, before
    /// any thread starts.
    ImageWriter(WriterOptions options, BoundedQueue<FrameResult>& input);

    ImageWriter(const ImageWriter&) = delete;
    ImageWriter& operator=(const ImageWriter&) = delete;

    void start();
    void join();

    /// Persists one result. Called by the writer thread, and directly by the
    /// sequential baseline, which has no queues.
    void write(FrameResult& result);

    /// Flushes and closes the metrics file. Idempotent.
    void finish();

    [[nodiscard]] std::size_t images_written() const noexcept { return images_written_; }
    [[nodiscard]] std::size_t write_failures() const noexcept { return write_failures_; }
    [[nodiscard]] const std::vector<double>& total_latencies_ms() const noexcept { return latencies_ms_; }
    [[nodiscard]] const std::vector<double>& process_times_ms() const noexcept { return process_ms_; }

private:
    struct FileCloser {
        void operator()(std::FILE* f) const noexcept { std::fclose(f); }
    };

    void run();

    WriterOptions options_;
    BoundedQueue<FrameResult>& input_;
    std::unique_ptr<std::FILE, FileCloser> metrics_;
    PosixThread thread_;

    std::size_t images_written_ = 0;
    std::size_t write_failures_ = 0;
    std::vector<double> latencies_ms_;
    std::vector<double> process_ms_;
};

}  // namespace pipeline
