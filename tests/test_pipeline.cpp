// End-to-end tests for the full reader -> workers -> writer pipeline.
//
// The key property: the parallel pipeline must produce exactly the same
// per-image results as the single-threaded baseline, for any worker count and
// queue capacity. ORB is deterministic, so any data race, dropped frame or
// duplicated frame shows up as a mismatch.

#include "Pipeline.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using pipeline::Pipeline;
using pipeline::PipelineConfig;
using pipeline::PipelineStats;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "  CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

struct TempDir {
    fs::path path;
    TempDir() : path(fs::temp_directory_path() / ("pipeline_test_" + std::to_string(::getpid()))) {
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

/// Writes `count` small textured images (plus one corrupt file) to `dir`.
void make_dataset(const fs::path& dir, int count) {
    fs::create_directories(dir / "nested");
    cv::RNG rng(1234);
    for (int i = 0; i < count; ++i) {
        cv::Mat img(240, 320, CV_8UC3);
        rng.fill(img, cv::RNG::UNIFORM, 0, 60);
        for (int s = 0; s < 25; ++s) {
            const cv::Point a(rng.uniform(0, 320), rng.uniform(0, 240));
            const cv::Point b(rng.uniform(0, 320), rng.uniform(0, 240));
            const cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            if (s % 2 == 0) {
                cv::rectangle(img, a, b, color, cv::FILLED);
            } else {
                cv::circle(img, a, rng.uniform(5, 40), color, 2);
            }
        }
        char name[32];
        std::snprintf(name, sizeof(name), "img_%03d.png", i);
        const fs::path target = (i % 5 == 0) ? dir / "nested" / name : dir / name;
        cv::imwrite(target.string(), img);
    }
    std::ofstream(dir / "corrupt.jpg") << "this is not a jpeg";
}

/// filename -> num_keypoints, parsed from a metrics CSV.
std::map<std::string, int> read_keypoints(const fs::path& csv, std::size_t& rows, std::set<std::string>& duplicates) {
    std::map<std::string, int> keypoints;
    std::ifstream in(csv);
    std::string line;
    std::getline(in, line);  // header
    rows = 0;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string sequence, filename, worker, width, height, kps;
        std::getline(ss, sequence, ',');
        std::getline(ss, filename, ',');
        std::getline(ss, worker, ',');
        std::getline(ss, width, ',');
        std::getline(ss, height, ',');
        std::getline(ss, kps, ',');
        if (!keypoints.emplace(filename, std::stoi(kps)).second) {
            duplicates.insert(filename);
        }
        ++rows;
    }
    return keypoints;
}

PipelineStats run(const fs::path& input, const fs::path& metrics, std::size_t workers, std::size_t queue,
                  bool sequential, const fs::path& output = {}) {
    PipelineConfig config;
    config.input_dir = input;
    config.metrics_path = metrics;
    config.num_workers = workers;
    config.queue_capacity = queue;
    config.sequential = sequential;
    if (!output.empty()) {
        config.output_dir = output;
        config.feature_maps = true;
    }
    Pipeline pipeline(config);
    return pipeline.run();
}

}  // namespace

int main() {
    cv::setNumThreads(0);
    TempDir tmp;
    const fs::path input = tmp.path / "input";
    constexpr int kImages = 40;
    make_dataset(input, kImages);

    // Baseline.
    const PipelineStats seq = run(input, tmp.path / "seq.csv", 1, 1, true);
    CHECK(seq.mode == "sequential");
    CHECK(seq.images_found == kImages + 1);
    CHECK(seq.images_processed == kImages);
    CHECK(seq.read_failures == 1);  // corrupt.jpg is skipped, not fatal
    std::size_t seq_rows = 0;
    std::set<std::string> seq_dups;
    const auto expected = read_keypoints(tmp.path / "seq.csv", seq_rows, seq_dups);
    CHECK(seq_rows == kImages);
    CHECK(seq_dups.empty());
    std::printf("[ OK ] sequential baseline: %zu images\n", seq.images_processed);

    // Parallel runs across worker counts and queue sizes, including the
    // degenerate capacity-1 queue that maximises contention and blocking.
    struct Case {
        std::size_t workers, queue;
    };
    for (const Case c : {Case{1, 1}, Case{2, 4}, Case{4, 1}, Case{8, 2}, Case{8, 32}, Case{16, 3}}) {
        const int before = g_failures;
        const fs::path csv = tmp.path / ("par_" + std::to_string(c.workers) + "_" + std::to_string(c.queue) + ".csv");
        const PipelineStats par = run(input, csv, c.workers, c.queue, false);
        std::size_t rows = 0;
        std::set<std::string> dups;
        const auto got = read_keypoints(csv, rows, dups);
        CHECK(par.images_processed == kImages);
        CHECK(par.read_failures == 1);
        CHECK(rows == kImages);
        CHECK(dups.empty());
        CHECK(got == expected);  // identical keypoint counts for every file
        std::size_t distributed = 0;
        for (std::size_t n : par.per_worker_counts) distributed += n;
        CHECK(distributed == kImages);
        std::printf("[%s] parallel workers=%-2zu queue=%-2zu matches baseline\n", g_failures == before ? " OK " : "FAIL",
                    c.workers, c.queue);
    }

    // Output images and feature maps are written for every frame.
    {
        const int before = g_failures;
        const fs::path out = tmp.path / "out";
        const PipelineStats par = run(input, tmp.path / "out.csv", 4, 4, false, out);
        CHECK(par.write_failures == 0);
        std::size_t keypoint_images = 0, feature_maps = 0;
        for (const auto& entry : fs::directory_iterator(out)) {
            const std::string name = entry.path().filename().string();
            keypoint_images += name.find("_keypoints.jpg") != std::string::npos;
            feature_maps += name.find("_featuremap.jpg") != std::string::npos;
        }
        CHECK(keypoint_images == kImages);
        CHECK(feature_maps == kImages);
        CHECK(fs::exists(out / "nested_img_000_keypoints.jpg"));  // nested path flattened
        std::printf("[%s] annotated images and feature maps written\n", g_failures == before ? " OK " : "FAIL");
    }

    // A stop requested before start processes nothing and terminates cleanly.
    {
        const int before = g_failures;
        PipelineConfig config;
        config.input_dir = input;
        config.metrics_path = tmp.path / "stopped.csv";
        config.num_workers = 4;
        Pipeline pipeline(config);
        pipeline.request_stop();
        const PipelineStats stats = pipeline.run();
        CHECK(stats.interrupted);
        CHECK(stats.images_processed == 0);
        std::printf("[%s] stop before start shuts down cleanly\n", g_failures == before ? " OK " : "FAIL");
    }

    if (g_failures != 0) {
        std::printf("\n%d check(s) FAILED\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("\nall pipeline tests passed\n");
    return EXIT_SUCCESS;
}
