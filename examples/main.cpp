#include "engine/core/vector_engine.hpp"
#include "engine/index/flat_index.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>

int main() {
    const std::filesystem::path wal_path = std::filesystem::temp_directory_path() / "example_engine.wal";
    std::filesystem::remove(wal_path);

    constexpr std::size_t dim = 4;
    engine::VectorEngine db(std::make_unique<engine::FlatIndex>(dim, engine::MetricType::L2), wal_path.string());

    const std::array<float, dim> a{1.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float, dim> b{0.0f, 1.0f, 0.0f, 0.0f};
    const std::array<float, dim> c{0.9f, 0.1f, 0.0f, 0.0f};
    db.insert(1, a);
    db.insert(2, b);
    db.insert(3, c);

    const auto results = db.search(a, 2);
    for (const auto& r : results) {
        std::printf("id=%llu distance=%f\n", static_cast<unsigned long long>(r.id), static_cast<double>(r.distance));
    }
    return 0;
}
