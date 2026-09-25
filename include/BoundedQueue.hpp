#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace pipeline {

/// A generic, thread-safe, bounded blocking FIFO queue.
///
/// * Backpressure: push() blocks while the queue is at capacity, so a fast
///   producer can never outrun slow consumers and exhaust memory.
/// * Graceful shutdown: close() wakes every blocked producer and consumer.
///   After close(), push() is rejected, while pop() keeps returning the items
///   that are still buffered and only then returns std::nullopt. That makes
///   "closed and drained" the unambiguous end-of-stream signal for consumers.
/// * Items are moved in and out, never copied, so move-only types such as
///   std::unique_ptr<T> work, and cv::Mat headers are transferred without
///   touching the pixel buffer.
///
/// Every piece of shared state is guarded by a single mutex. Two condition
/// variables separate the "space available" and "item available" wake-ups so
/// that producers only wake consumers and vice versa.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("BoundedQueue capacity must be > 0");
        }
    }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;
    BoundedQueue(BoundedQueue&&) = delete;
    BoundedQueue& operator=(BoundedQueue&&) = delete;
    ~BoundedQueue() = default;

    /// Blocks while the queue is full. Returns false (and drops `item`) if the
    /// queue is closed before space becomes available.
    bool push(T item) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            not_full_cv_.wait(lock, [this] { return closed_ || items_.size() < capacity_; });
            if (closed_) {
                return false;
            }
            items_.push_back(std::move(item));
        }
        // Notify outside the lock so the woken consumer does not immediately
        // block on a mutex we still hold.
        not_empty_cv_.notify_one();
        return true;
    }

    /// Non-blocking push. Returns false if the queue is full or closed.
    bool try_push(T& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_ || items_.size() >= capacity_) {
                return false;
            }
            items_.push_back(std::move(item));
        }
        not_empty_cv_.notify_one();
        return true;
    }

    /// Blocks while the queue is empty and open. Returns std::nullopt only
    /// once the queue has been closed *and* fully drained.
    std::optional<T> pop() {
        std::optional<T> item;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            not_empty_cv_.wait(lock, [this] { return closed_ || !items_.empty(); });
            if (items_.empty()) {
                return std::nullopt;  // closed and drained
            }
            item.emplace(std::move(items_.front()));
            items_.pop_front();
        }
        not_full_cv_.notify_one();
        return item;
    }

    /// Non-blocking pop. Returns std::nullopt if the queue is currently empty.
    std::optional<T> try_pop() {
        std::optional<T> item;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (items_.empty()) {
                return std::nullopt;
            }
            item.emplace(std::move(items_.front()));
            items_.pop_front();
        }
        not_full_cv_.notify_one();
        return item;
    }

    /// Idempotent. Rejects future pushes and wakes every waiting thread.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_full_cv_.notify_all();
        not_empty_cv_.notify_all();
    }

    [[nodiscard]] bool closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return items_.size();
    }

    [[nodiscard]] bool empty() const { return size() == 0; }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
    std::deque<T> items_;
    bool closed_ = false;
};

}  // namespace pipeline
