#include "ImageWriter.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace pipeline {

namespace fs = std::filesystem;

namespace {

constexpr const char* kCsvHeader =
    "sequence,filename,worker_id,width,height,num_keypoints,"
    "read_time_ms,queue_wait_ms,process_time_ms,output_wait_ms,write_time_ms,total_latency_ms,"
    "sharpness,brightness,contrast\n";

/// RFC 4180 quoting: only wraps the field when it needs it.
std::string csv_field(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string quoted = "\"";
    for (char c : value) {
        if (c == '"') {
            quoted += '"';
        }
        quoted += c;
    }
    quoted += '"';
    return quoted;
}

/// "sub/dir/img.png" -> "sub_dir_img" so nested datasets map to a flat output dir.
std::string flat_stem(const std::string& relative_name) {
    std::string stem = fs::path(relative_name).replace_extension().generic_string();
    std::replace(stem.begin(), stem.end(), '/', '_');
    return stem;
}

}  // namespace

ImageWriter::ImageWriter(WriterOptions options, BoundedQueue<FrameResult>& input)
    : options_(std::move(options)), input_(input) {
    if (options_.output_dir) {
        std::error_code ec;
        fs::create_directories(*options_.output_dir, ec);
        if (ec) {
            throw std::runtime_error("cannot create output directory " + options_.output_dir->string() + ": " +
                                     ec.message());
        }
    }
    if (options_.metrics_path.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(options_.metrics_path.parent_path(), ec);
    }
    metrics_.reset(std::fopen(options_.metrics_path.string().c_str(), "w"));
    if (!metrics_) {
        throw std::runtime_error("cannot open metrics file " + options_.metrics_path.string() + ": " +
                                 std::strerror(errno));
    }
    std::fputs(kCsvHeader, metrics_.get());
}

void ImageWriter::start() {
    thread_.start("writer", [this] { run(); });
}

void ImageWriter::join() { thread_.join(); }

void ImageWriter::write(FrameResult& result) {
    const TimePoint start = Clock::now();
    const double output_wait_ms = elapsed_ms(result.processed_timestamp, start);

    if (options_.output_dir) {
        const std::string stem = flat_stem(result.filename);
        const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, options_.jpeg_quality};
        bool ok = cv::imwrite((*options_.output_dir / (stem + "_keypoints.jpg")).string(), result.annotated_image,
                              params);
        if (!result.feature_map.empty()) {
            ok = cv::imwrite((*options_.output_dir / (stem + "_featuremap.jpg")).string(), result.feature_map,
                             params) &&
                 ok;
        }
        if (!ok) {
            ++write_failures_;
        }
    }
    // Release the pixel buffers as early as possible.
    result.annotated_image.release();
    result.feature_map.release();

    const TimePoint end = Clock::now();
    result.write_time_ms = elapsed_ms(start, end);
    result.total_latency_ms = elapsed_ms(result.read_start, end);

    std::fprintf(metrics_.get(), "%llu,%s,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                 static_cast<unsigned long long>(result.sequence), csv_field(result.filename).c_str(),
                 result.worker_id, result.width, result.height, result.num_keypoints, result.read_time_ms,
                 result.queue_wait_ms, result.process_time_ms, output_wait_ms, result.write_time_ms,
                 result.total_latency_ms, result.sharpness, result.brightness, result.contrast);

    latencies_ms_.push_back(result.total_latency_ms);
    process_ms_.push_back(result.process_time_ms);
    ++images_written_;
}

void ImageWriter::finish() {
    if (metrics_) {
        std::fflush(metrics_.get());
        metrics_.reset();
    }
}

void ImageWriter::run() {
    while (std::optional<FrameResult> result = input_.pop()) {
        write(*result);
    }
    finish();
}

}  // namespace pipeline
