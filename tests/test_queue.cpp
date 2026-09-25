// Stress and semantics tests for pipeline::BoundedQueue.
//
// Self-contained (no test framework) so it builds anywhere and runs cleanly
// under ThreadSanitizer and AddressSanitizer. Every test runs under a
// watchdog: if the test does not finish in time it is reported as a deadlock
// and the process aborts, instead of hanging CI forever.

#include "BoundedQueue.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using pipeline::BoundedQueue;
using namespace std::chrono_literals;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "  CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

/// Aborts the process if `body` has not returned within `timeout`.
void with_watchdog(const char* name, std::chrono::seconds timeout, const std::function<void()>& body) {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::thread watchdog([&] {
        std::unique_lock<std::mutex> lock(m);
        if (!cv.wait_for(lock, timeout, [&] { return done; })) {
            std::fprintf(stderr, "DEADLOCK: test '%s' did not finish within %llds\n", name,
                         static_cast<long long>(timeout.count()));
            std::abort();
        }
    });
    const int failures_before = g_failures;
    const auto start = std::chrono::steady_clock::now();
    body();
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    {
        std::lock_guard<std::mutex> lock(m);
        done = true;
    }
    cv.notify_one();
    watchdog.join();
    std::printf("[%s] %-48s %9.1f ms\n", g_failures == failures_before ? " OK " : "FAIL", name, ms);
}

// ---------------------------------------------------------------------------

void test_fifo_single_thread() {
    BoundedQueue<int> q(4);
    CHECK(q.capacity() == 4);
    CHECK(q.empty());
    for (int i = 0; i < 4; ++i) {
        CHECK(q.push(i));
    }
    CHECK(q.size() == 4);
    int blocked_item = 99;
    CHECK(!q.try_push(blocked_item));  // full
    for (int i = 0; i < 4; ++i) {
        auto v = q.pop();
        CHECK(v.has_value() && *v == i);
    }
    CHECK(!q.try_pop().has_value());
}

void test_invalid_capacity() {
    bool threw = false;
    try {
        BoundedQueue<int> q(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_move_only_type() {
    BoundedQueue<std::unique_ptr<int>> q(2);
    CHECK(q.push(std::make_unique<int>(7)));
    CHECK(q.push(std::make_unique<int>(8)));
    q.close();
    auto a = q.pop();
    auto b = q.pop();
    CHECK(a && *a && **a == 7);
    CHECK(b && *b && **b == 8);
    CHECK(!q.pop().has_value());
}

void test_close_drains_then_ends() {
    BoundedQueue<int> q(8);
    for (int i = 0; i < 5; ++i) {
        CHECK(q.push(i));
    }
    q.close();
    q.close();  // idempotent
    CHECK(q.closed());
    CHECK(!q.push(100));  // rejected after close
    for (int i = 0; i < 5; ++i) {
        auto v = q.pop();
        CHECK(v.has_value() && *v == i);  // buffered items survive close
    }
    CHECK(!q.pop().has_value());
    CHECK(!q.pop().has_value());  // stays at end-of-stream
}

void test_close_wakes_blocked_consumers() {
    BoundedQueue<int> q(4);
    constexpr int kConsumers = 8;
    std::atomic<int> woke_empty{0};
    std::vector<std::thread> consumers;
    for (int i = 0; i < kConsumers; ++i) {
        consumers.emplace_back([&] {
            if (!q.pop().has_value()) {
                woke_empty.fetch_add(1);
            }
        });
    }
    std::this_thread::sleep_for(50ms);  // let them block on the empty queue
    q.close();
    for (auto& t : consumers) {
        t.join();
    }
    CHECK(woke_empty.load() == kConsumers);
}

void test_close_wakes_blocked_producers() {
    BoundedQueue<int> q(2);
    CHECK(q.push(1));
    CHECK(q.push(2));
    constexpr int kProducers = 8;
    std::atomic<int> rejected{0};
    std::vector<std::thread> producers;
    for (int i = 0; i < kProducers; ++i) {
        producers.emplace_back([&, i] {
            if (!q.push(100 + i)) {
                rejected.fetch_add(1);
            }
        });
    }
    std::this_thread::sleep_for(50ms);  // let them block on the full queue
    q.close();
    for (auto& t : producers) {
        t.join();
    }
    CHECK(rejected.load() == kProducers);
    CHECK(q.size() == 2);
}

/// A producer must block at capacity and resume as soon as space frees up.
void test_backpressure() {
    constexpr std::size_t kCapacity = 16;
    BoundedQueue<int> q(kCapacity);
    std::atomic<int> pushed{0};
    std::thread producer([&] {
        for (int i = 0; i < 100; ++i) {
            if (!q.push(i)) {
                return;
            }
            pushed.fetch_add(1);
        }
    });
    std::this_thread::sleep_for(100ms);
    CHECK(pushed.load() == static_cast<int>(kCapacity));  // stalled at the bound
    CHECK(q.size() == kCapacity);

    auto first = q.pop();
    CHECK(first && *first == 0);
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (pushed.load() < static_cast<int>(kCapacity) + 1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    CHECK(pushed.load() == static_cast<int>(kCapacity) + 1);  // exactly one slot opened
    CHECK(q.size() <= kCapacity);

    q.close();
    producer.join();
}

/// Many producers, one consumer: items from each producer arrive in the order
/// that producer pushed them.
void test_per_producer_fifo() {
    constexpr int kProducers = 8;
    constexpr int kPerProducer = 5000;
    BoundedQueue<std::pair<int, int>> q(32);
    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
            for (int i = 0; i < kPerProducer; ++i) {
                q.push({p, i});
            }
        });
    }
    std::vector<int> next(kProducers, 0);
    bool ordered = true;
    std::thread consumer([&] {
        while (auto item = q.pop()) {
            if (item->second != next[static_cast<std::size_t>(item->first)]) {
                ordered = false;
            }
            ++next[static_cast<std::size_t>(item->first)];
        }
    });
    for (auto& t : producers) {
        t.join();
    }
    q.close();
    consumer.join();
    CHECK(ordered);
    for (int p = 0; p < kProducers; ++p) {
        CHECK(next[static_cast<std::size_t>(p)] == kPerProducer);
    }
}

/// The headline test: 16 producers and 16 consumers (32 threads) move 100,000
/// uniquely numbered items through the queue. Every item must be delivered
/// exactly once: nothing dropped, nothing duplicated, and every thread
/// terminates.
void test_mpmc_stress(std::size_t capacity) {
    constexpr int kProducers = 16;
    constexpr int kConsumers = 16;
    constexpr std::uint32_t kItems = 100'000;
    constexpr std::uint32_t kPerProducer = kItems / kProducers;
    static_assert(kItems % kProducers == 0, "items must divide evenly");

    BoundedQueue<std::uint32_t> q(capacity);
    std::vector<std::vector<std::uint32_t>> received(kConsumers);
    std::vector<std::thread> threads;
    std::atomic<bool> go{false};

    for (int c = 0; c < kConsumers; ++c) {
        threads.emplace_back([&, c] {
            while (!go.load()) std::this_thread::yield();
            auto& mine = received[static_cast<std::size_t>(c)];
            while (auto v = q.pop()) {
                mine.push_back(*v);
            }
        });
    }
    std::vector<std::thread> producers;
    std::atomic<int> rejected{0};
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
            while (!go.load()) std::this_thread::yield();
            const std::uint32_t base = static_cast<std::uint32_t>(p) * kPerProducer;
            for (std::uint32_t i = 0; i < kPerProducer; ++i) {
                if (!q.push(base + i)) {
                    rejected.fetch_add(1);
                }
            }
        });
    }
    go.store(true);
    for (auto& t : producers) {
        t.join();
    }
    q.close();  // all producers done: end of stream
    for (auto& t : threads) {
        t.join();
    }

    CHECK(rejected.load() == 0);
    std::vector<std::uint8_t> seen(kItems, 0);
    std::size_t total = 0;
    bool in_range = true;
    for (const auto& vec : received) {
        total += vec.size();
        for (std::uint32_t v : vec) {
            if (v >= kItems) {
                in_range = false;
                continue;
            }
            ++seen[v];
        }
    }
    CHECK(in_range);
    CHECK(total == kItems);
    CHECK(std::all_of(seen.begin(), seen.end(), [](std::uint8_t n) { return n == 1; }));
    CHECK(q.empty());
}

/// Shutdown in the middle of heavy traffic: consumers stop early, the queue
/// is closed while producers are still blocked. Every accepted push must still
/// be accounted for by a pop, and every thread must exit.
void test_close_under_load() {
    constexpr int kProducers = 8;
    constexpr int kConsumers = 8;
    BoundedQueue<int> q(8);
    std::atomic<long> accepted{0};
    std::atomic<long> consumed{0};
    std::vector<std::thread> threads;
    for (int p = 0; p < kProducers; ++p) {
        threads.emplace_back([&] {
            for (int i = 0;; ++i) {
                if (!q.push(i)) {
                    return;
                }
                accepted.fetch_add(1);
            }
        });
    }
    for (int c = 0; c < kConsumers; ++c) {
        threads.emplace_back([&] {
            while (q.pop()) {
                consumed.fetch_add(1);
            }
        });
    }
    std::this_thread::sleep_for(100ms);
    q.close();
    for (auto& t : threads) {
        t.join();
    }
    CHECK(accepted.load() > 0);
    CHECK(accepted.load() == consumed.load());
    CHECK(q.empty());
}

}  // namespace

int main() {
    std::printf("BoundedQueue tests (hardware threads: %u)\n", std::thread::hardware_concurrency());
    const auto timeout = 120s;  // generous: TSan slows execution 5-15x
    with_watchdog("fifo_single_thread", timeout, test_fifo_single_thread);
    with_watchdog("invalid_capacity", timeout, test_invalid_capacity);
    with_watchdog("move_only_type", timeout, test_move_only_type);
    with_watchdog("close_drains_then_ends", timeout, test_close_drains_then_ends);
    with_watchdog("close_wakes_blocked_consumers", timeout, test_close_wakes_blocked_consumers);
    with_watchdog("close_wakes_blocked_producers", timeout, test_close_wakes_blocked_producers);
    with_watchdog("backpressure", timeout, test_backpressure);
    with_watchdog("per_producer_fifo", timeout, test_per_producer_fifo);
    for (std::size_t capacity : {1u, 8u, 64u, 1024u}) {
        const std::string name = "mpmc_stress 16P/16C 100k items cap=" + std::to_string(capacity);
        with_watchdog(name.c_str(), timeout, [capacity] { test_mpmc_stress(capacity); });
    }
    with_watchdog("close_under_load", timeout, test_close_under_load);

    if (g_failures != 0) {
        std::printf("\n%d check(s) FAILED\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("\nall tests passed\n");
    return EXIT_SUCCESS;
}
