#pragma once

#include "BoundedQueue.hpp"
#include "PosixThread.hpp"
#include "Types.hpp"

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace pipeline {

/// Stage 1: a single producer thread that decodes images from disk and feeds
/// the input queue. Closes the queue when the listing is exhausted or a stop
/// is requested, which is the end-of-stream signal for the worker pool.
class ImageReader {
public:
    ImageReader(std::filesystem::path input_dir, std::vector<std::filesystem::path> files,
                BoundedQueue<FrameTask>& output, const std::atomic<bool>& stop_requested);

    void start();
    void join();

    [[nodiscard]] std::size_t images_read() const noexcept { return images_read_; }
    [[nodiscard]] std::size_t failures() const noexcept { return failures_; }

    /// Recursively lists supported image files under `dir`, sorted by path so
    /// that runs are reproducible and sequence numbers are stable.
    static std::vector<std::filesystem::path> list_images(const std::filesystem::path& dir);

    /// Reads and decodes one image. Returns std::nullopt if the file cannot be
    /// decoded. Shared by the threaded reader and the sequential baseline.
    static std::optional<FrameTask> load(const std::filesystem::path& path, std::string display_name,
                                         std::uint64_t sequence);

private:
    void run();

    std::filesystem::path input_dir_;
    std::vector<std::filesystem::path> files_;
    BoundedQueue<FrameTask>& output_;
    const std::atomic<bool>& stop_requested_;
    PosixThread thread_;

    // Written only by the reader thread; read by the coordinator after join(),
    // which provides the happens-before edge.
    std::size_t images_read_ = 0;
    std::size_t failures_ = 0;
};

}  // namespace pipeline
