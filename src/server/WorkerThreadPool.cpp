#include "WorkerThreadPool.hpp"

WorkerThreadPool::WorkerThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this]() {
            worker_loop();

        });

    }
}

WorkerThreadPool::~WorkerThreadPool() {
    {
        std::unique_lock lock(queue_mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }

    }

}

void WorkerThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock lock(queue_mutex_);
        tasks_.push(task);
    }
    cv_.notify_one();


}

void WorkerThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock lock(queue_mutex_);
            cv_.wait(lock, [this]() {
                return stop_ || !tasks_.empty();
            });

            if (tasks_.empty() && stop_) {
                return;
            }

            task = tasks_.front();
            tasks_.pop();
        }

        task();
    }
}