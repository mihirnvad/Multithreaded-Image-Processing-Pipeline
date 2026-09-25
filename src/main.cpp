#include "Pipeline.hpp"

#include <opencv2/core.hpp>

#include <getopt.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s --input <dir> [options]\n"
        "\n"
        "Multithreaded ORB feature-extraction pipeline:\n"
        "  reader thread -> bounded queue -> N worker threads -> bounded queue -> writer thread\n"
        "\n"
        "Options:\n"
        "  -i, --input <dir>        Input image directory, searched recursively (required)\n"
        "  -o, --output <dir>       Save annotated images (keypoints drawn in green) here\n"
        "  -t, --threads <n>        Worker threads (default: hardware concurrency = %u)\n"
        "  -q, --queue-size <n>     Capacity of each bounded queue (default: 32)\n"
        "  -m, --metrics <file>     Per-image metrics CSV (default: metrics.csv)\n"
        "  -f, --features <n>       ORB features per image (default: 1000)\n"
        "  -F, --feature-maps       Also save keypoint-density heat maps (needs --output)\n"
        "  -s, --sequential         Single-threaded baseline: no queues, no worker threads\n"
        "  -n, --limit <n>          Process at most n images\n"
        "  -j, --json               Print the run summary as a single JSON line\n"
        "  -h, --help               Show this help\n",
        argv0, std::thread::hardware_concurrency());
}

std::size_t parse_positive(const char* flag, const char* text) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value > std::numeric_limits<unsigned int>::max()) {
        throw std::invalid_argument(std::string("invalid value for ") + flag + ": '" + text + "'");
    }
    return static_cast<std::size_t>(value);
}

void print_human_summary(const pipeline::PipelineStats& s) {
    std::fprintf(stderr,
                 "\n[%s] workers=%zu images=%zu/%zu failed=%zu%s\n"
                 "  wall time      %10.1f ms\n"
                 "  throughput     %10.2f images/s\n"
                 "  mean process   %10.2f ms/image\n"
                 "  latency p50    %10.2f ms\n"
                 "  latency p95    %10.2f ms\n"
                 "  latency p99    %10.2f ms\n",
                 s.mode.c_str(), s.workers, s.images_processed, s.images_found, s.read_failures + s.write_failures,
                 s.interrupted ? " (interrupted)" : "", s.wall_time_ms, s.throughput_ips, s.mean_process_ms,
                 s.p50_latency_ms, s.p95_latency_ms, s.p99_latency_ms);
    if (s.per_worker_counts.size() > 1) {
        std::fprintf(stderr, "  per worker    ");
        for (std::size_t count : s.per_worker_counts) {
            std::fprintf(stderr, " %zu", count);
        }
        std::fprintf(stderr, "\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    pipeline::PipelineConfig config;
    config.num_workers = std::max(1u, std::thread::hardware_concurrency());
    bool json = false;

    static const option kLongOptions[] = {
        {"input", required_argument, nullptr, 'i'},    {"output", required_argument, nullptr, 'o'},
        {"threads", required_argument, nullptr, 't'},  {"queue-size", required_argument, nullptr, 'q'},
        {"metrics", required_argument, nullptr, 'm'},  {"features", required_argument, nullptr, 'f'},
        {"feature-maps", no_argument, nullptr, 'F'},   {"sequential", no_argument, nullptr, 's'},
        {"limit", required_argument, nullptr, 'n'},    {"json", no_argument, nullptr, 'j'},
        {"help", no_argument, nullptr, 'h'},           {nullptr, 0, nullptr, 0},
    };

    try {
        int opt = 0;
        while ((opt = getopt_long(argc, argv, "i:o:t:q:m:f:Fsn:jh", kLongOptions, nullptr)) != -1) {
            switch (opt) {
                case 'i': config.input_dir = optarg; break;
                case 'o': config.output_dir = std::filesystem::path(optarg); break;
                case 't': config.num_workers = parse_positive("--threads", optarg); break;
                case 'q': config.queue_capacity = parse_positive("--queue-size", optarg); break;
                case 'm': config.metrics_path = optarg; break;
                case 'f': config.max_features = static_cast<int>(parse_positive("--features", optarg)); break;
                case 'F': config.feature_maps = true; break;
                case 's': config.sequential = true; break;
                case 'n': config.limit = parse_positive("--limit", optarg); break;
                case 'j': json = true; break;
                case 'h': print_usage(argv[0]); return EXIT_SUCCESS;
                default: print_usage(argv[0]); return 2;
            }
        }
        if (optind < argc) {
            throw std::invalid_argument(std::string("unexpected argument: ") + argv[optind]);
        }
        if (config.input_dir.empty()) {
            throw std::invalid_argument("--input is required");
        }
        if (config.feature_maps && !config.output_dir) {
            std::fprintf(stderr, "warning: --feature-maps has no effect without --output\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n\n", e.what());
        print_usage(argv[0]);
        return 2;
    }

    // Parallelism comes from our own threads. OpenCV's internal thread pool is
    // disabled so it neither oversubscribes the cores nor silently speeds up
    // the "sequential" baseline, which would make the comparison meaningless.
    cv::setNumThreads(0);

    try {
        pipeline::Pipeline pipeline(config);
        // Installed before any worker thread exists so they all inherit the
        // blocked signal mask (see ScopedSignalHandler).
        pipeline::ScopedSignalHandler signals([&pipeline](int signo) {
            std::fprintf(stderr, "\nreceived %s, finishing in-flight images (press Ctrl+C again to abort)\n",
                         signo == SIGINT ? "SIGINT" : "SIGTERM");
            pipeline.request_stop();
        });

        const pipeline::PipelineStats stats = pipeline.run();
        print_human_summary(stats);
        if (json) {
            std::printf("%s\n", stats.to_json().c_str());
        }
        if (stats.interrupted) {
            return 130;
        }
        return stats.read_failures + stats.write_failures > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
