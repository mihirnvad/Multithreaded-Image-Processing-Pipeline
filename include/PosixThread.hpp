#pragma once

#include <pthread.h>

#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace pipeline {

/// Minimal RAII owner of a POSIX thread (pthread_create / pthread_join).
///
/// Semantics mirror std::thread with two differences that matter here:
///  * the thread can be given a human-readable name that shows up in
///    `top -H`, `htop`, gdb and perf (pthread_setname_np, Linux only);
///  * the destructor joins instead of calling std::terminate, so a thread can
///    never be leaked if an exception unwinds past its owner.
class PosixThread {
public:
    PosixThread() = default;

    PosixThread(std::string name, std::function<void()> body) { start(std::move(name), std::move(body)); }

    PosixThread(const PosixThread&) = delete;
    PosixThread& operator=(const PosixThread&) = delete;

    PosixThread(PosixThread&& other) noexcept
        : handle_(other.handle_), joinable_(std::exchange(other.joinable_, false)) {}

    PosixThread& operator=(PosixThread&& other) noexcept {
        if (this != &other) {
            join();
            handle_ = other.handle_;
            joinable_ = std::exchange(other.joinable_, false);
        }
        return *this;
    }

    ~PosixThread() { join(); }

    void start(std::string name, std::function<void()> body) {
        if (joinable_) {
            throw std::logic_error("PosixThread::start called on a running thread");
        }
        auto ctx = std::make_unique<Context>(Context{std::move(name), std::move(body)});
        const int rc = pthread_create(&handle_, nullptr, &PosixThread::trampoline, ctx.get());
        if (rc != 0) {
            throw std::system_error(rc, std::generic_category(), "pthread_create");
        }
        ctx.release();  // ownership passed to the new thread
        joinable_ = true;
    }

    void join() {
        if (!joinable_) {
            return;
        }
        joinable_ = false;
        pthread_join(handle_, nullptr);
    }

    [[nodiscard]] bool joinable() const noexcept { return joinable_; }
    [[nodiscard]] pthread_t native_handle() const noexcept { return handle_; }

private:
    struct Context {
        std::string name;
        std::function<void()> body;
    };

    static void* trampoline(void* arg) {
        std::unique_ptr<Context> ctx(static_cast<Context*>(arg));
#if defined(__linux__)
        // Linux limits thread names to 15 characters plus the terminator.
        pthread_setname_np(pthread_self(), ctx->name.substr(0, 15).c_str());
#endif
        try {
            ctx->body();
        } catch (const std::exception& e) {
            // An exception escaping a thread would otherwise terminate the
            // process with no context. Report it and terminate explicitly.
            std::fprintf(stderr, "fatal: uncaught exception in thread '%s': %s\n", ctx->name.c_str(), e.what());
            std::terminate();
        }
        return nullptr;
    }

    pthread_t handle_{};
    bool joinable_ = false;
};

}  // namespace pipeline
