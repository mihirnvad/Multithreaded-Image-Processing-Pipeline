#include "ImageReader.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <string_view>
#include <system_error>
#include <utility>

namespace pipeline {

namespace fs = std::filesystem;

namespace {

bool has_image_extension(const fs::path& path) {
    static constexpr std::array<std::string_view, 9> kExtensions = {".jpg", ".jpeg", ".png",  ".bmp", ".tif",
                                                                     ".tiff", ".webp", ".ppm", ".pgm"};
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return std::find(kExtensions.begin(), kExtensions.end(), ext) != kExtensions.end();
}

}  // namespace

ImageReader::ImageReader(fs::path input_dir, std::vector<fs::path> files, BoundedQueue<FrameTask>& output,
                         const std::atomic<bool>& stop_requested)
    : input_dir_(std::move(input_dir)),
      files_(std::move(files)),
      output_(output),
      stop_requested_(stop_requested) {}

void ImageReader::start() {
    thread_.start("reader", [this] { run(); });
}

void ImageReader::join() { thread_.join(); }

std::vector<fs::path> ImageReader::list_images(const fs::path& dir) {
    std::vector<fs::path> files;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec) && has_image_extension(it->path())) {
            files.push_back(it->path());
        }
    }
    if (ec) {
        throw fs::filesystem_error("cannot list input directory", dir, ec);
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::optional<FrameTask> ImageReader::load(const fs::path& path, std::string display_name, std::uint64_t sequence) {
    FrameTask task;
    task.sequence = sequence;
    task.filename = std::move(display_name);
    task.read_start = Clock::now();
    task.raw_image = cv::imread(path.string(), cv::IMREAD_COLOR);
    task.read_timestamp = Clock::now();
    task.read_time_ms = elapsed_ms(task.read_start, task.read_timestamp);
    if (task.raw_image.empty()) {
        return std::nullopt;
    }
    return task;
}

void ImageReader::run() {
    std::uint64_t sequence = 0;
    for (const fs::path& path : files_) {
        if (stop_requested_.load(std::memory_order_relaxed)) {
            break;
        }
        std::optional<FrameTask> task = load(path, path.lexically_relative(input_dir_).generic_string(), sequence++);
        if (!task) {
            ++failures_;
            std::fprintf(stderr, "warning: could not decode %s, skipping\n", path.string().c_str());
            continue;
        }
        // Blocks while the queue is full: this is the backpressure that caps
        // memory at queue_capacity decoded frames regardless of dataset size.
        if (!output_.push(std::move(*task))) {
            break;  // queue closed underneath us (shutdown)
        }
        ++images_read_;
    }
    output_.close();
}

}  // namespace pipeline
