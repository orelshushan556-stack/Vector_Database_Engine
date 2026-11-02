#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

class WorkerThreadPool {
public:
    explicit WorkerThreadPool(size_t num_threads);
    ~WorkerThreadPool();

    WorkerThreadPool(const WorkerThreadPool&) = delete;
    WorkerThreadPool& operator=(const WorkerThreadPool&) = delete;
    WorkerThreadPool(WorkerThreadPool&&) = delete;
    WorkerThreadPool& operator=(WorkerThreadPool&&) = delete;

    void enqueue(std::function<void()> task);

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
};