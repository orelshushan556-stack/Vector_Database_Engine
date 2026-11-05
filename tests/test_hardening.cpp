// Regression tests for defects found by a systematic audit: input validation at trust
// boundaries, undefined behavior on non-finite floats, and corrupt snapshot handling.

#include <gtest/gtest.h>

#include "engine/core/vector_engine.hpp"
#include "engine/index/flat_index.hpp"
#include "engine/index/hnsw_index.hpp"
#include "engine/index/hnsw_index_sq8.hpp"
#include "engine/quantization/scalar_quantizer.hpp"
#include "engine/storage/serializer.hpp"

#include "HttpRequestParser.hpp"
#include "JsonUtils.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace engine;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

std::vector<float> random_data(std::size_t count, std::size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);
    return data;
}

// ------------------------------------------------------------------ engine boundary

TEST(HardeningTest, EngineRejectsNonFiniteVectorsBeforeTouchingTheWal) {
    const std::string wal_path = "hardening_engine.wal";
    fs::remove(wal_path);
    {
        VectorEngine eng(std::make_unique<FlatIndex>(4, MetricType::L2), wal_path);

        const std::vector<float> nan_vec = {1.0f, kNaN, 3.0f, 4.0f};
        const std::vector<float> inf_vec = {1.0f, 2.0f, kInf, 4.0f};
        EXPECT_THROW(eng.insert(1, nan_vec), std::invalid_argument);
        EXPECT_THROW(eng.insert(2, inf_vec), std::invalid_argument);
        EXPECT_THROW(static_cast<void>(eng.search(nan_vec, 3)), std::invalid_argument);
        EXPECT_EQ(eng.size(), 0u);
    }
    EXPECT_EQ(fs::file_size(wal_path), 0u);
    fs::remove(wal_path);
}

// ------------------------------------------------------------------ quantizer

TEST(HardeningTest, QuantizerRejectsNonFiniteTrainingDataAndEncodesNonFiniteSafely) {
    ScalarQuantizer bad(2);
    const std::vector<float> bad_data = {0.0f, 1.0f, kNaN, 2.0f};
    EXPECT_THROW(bad.train(bad_data), std::invalid_argument);

    ScalarQuantizer q(2);
    const std::vector<float> data = {0.0f, 0.0f, 10.0f, 10.0f};
    q.train(data);

    const std::vector<float> weird = {kNaN, kInf};
    std::vector<uint8_t> codes(2, 99);
    EXPECT_NO_THROW(q.encode(weird, codes));  // converting NaN to uint8_t would be UB
    EXPECT_EQ(codes[0], 0);
    EXPECT_EQ(codes[1], 255);
}

// ------------------------------------------------------------------ constructors

TEST(HardeningTest, HnswConstructorsRejectDegenerateParameters) {
    EXPECT_THROW(HNSWIndex(8, MetricType::L2, 1), std::invalid_argument);
    EXPECT_THROW(HNSWIndex(8, MetricType::L2, 0), std::invalid_argument);
    EXPECT_THROW(HNSWIndex(0, MetricType::L2, 16), std::invalid_argument);

    ScalarQuantizer q(4);
    const std::vector<float> data = {0, 0, 0, 0, 1, 1, 1, 1};
    q.train(data);
    EXPECT_THROW(HNSWIndexSQ8(4, MetricType::L2, q, 1), std::invalid_argument);
    EXPECT_THROW(HNSWIndexSQ8(4, MetricType::L2, q, 0), std::invalid_argument);
}

TEST(HardeningTest, Sq8SearchToleratesDegenerateReRankFactors) {
    constexpr std::size_t dim = 8, n = 50;
    const auto data = random_data(n, dim, 1);
    ScalarQuantizer q(dim);
    q.train(data);
    HNSWIndexSQ8 index(dim, MetricType::L2, q, 8, 50, 20);
    for (std::size_t i = 0; i < n; ++i) index.add(i, VectorView(data.data() + i * dim, dim));

    const VectorView query(data.data(), dim);
    for (float factor : {kNaN, -3.0f, 0.0f, 1e30f, kInf}) {
        index.set_re_rank_factor(factor);
        EXPECT_NO_THROW({
            const auto res = index.search(query, 5);
            EXPECT_LE(res.size(), 5u);
        });
    }
}

TEST(HardeningTest, HugeEfAndKDoNotSizeBuffersFromTheCaller) {
    constexpr std::size_t dim = 8, n = 50;
    const auto data = random_data(n, dim, 2);
    HNSWIndex index(dim, MetricType::L2, 8, 50, std::numeric_limits<std::size_t>::max() / 4);
    for (std::size_t i = 0; i < n; ++i) index.add(i, VectorView(data.data() + i * dim, dim));
    const auto res = index.search(VectorView(data.data(), dim), std::numeric_limits<std::size_t>::max() / 8);
    EXPECT_EQ(res.size(), n);
}

// ------------------------------------------------------------------ snapshot loader

std::string serialize_to_string(const IndexBase& index) {
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    if (const auto* flat = dynamic_cast<const FlatIndex*>(&index)) {
        Serializer::serialize(*flat, ss);
    } else if (const auto* h = dynamic_cast<const HNSWIndex*>(&index)) {
        Serializer::serialize(*h, ss);
    } else if (const auto* s = dynamic_cast<const HNSWIndexSQ8*>(&index)) {
        Serializer::serialize(*s, ss);
    }
    return ss.str();
}

std::unique_ptr<IndexBase> try_load(const std::string& bytes) {
    std::stringstream ss(bytes, std::ios::in | std::ios::binary);
    return Serializer::load(ss);
}

TEST(HardeningTest, TruncatedSnapshotsAreRejectedAtEveryLength) {
    constexpr std::size_t dim = 4, n = 6;
    const auto data = random_data(n, dim, 3);
    FlatIndex flat(dim, MetricType::L2);
    for (std::size_t i = 0; i < n; ++i) flat.add(i, VectorView(data.data() + i * dim, dim));

    const std::string bytes = serialize_to_string(flat);
    ASSERT_NO_THROW(try_load(bytes));
    for (std::size_t len = 0; len < bytes.size(); ++len) {
        EXPECT_THROW(try_load(bytes.substr(0, len)), std::runtime_error) << "prefix length " << len;
    }
}

TEST(HardeningTest, SnapshotWithHugeLengthFieldFailsWithoutHugeAllocation) {
    FileHeader header;
    header.dimension = 4;
    header.count = std::uint64_t{1} << 40;  // would need terabytes if trusted
    std::string bytes(reinterpret_cast<const char*>(&header), sizeof(header));
    bytes.append(64, '\0');
    EXPECT_THROW(try_load(bytes), std::runtime_error);

    header.count = std::numeric_limits<std::uint64_t>::max();  // count * dimension overflows
    bytes.assign(reinterpret_cast<const char*>(&header), sizeof(header));
    EXPECT_THROW(try_load(bytes), std::runtime_error);
}

// Flips random bytes in valid snapshots. Every outcome must be either a clean exception or
// an index that can be searched without touching memory out of bounds (checked by ASan).
void fuzz_snapshot(const std::string& valid, std::size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::size_t rejected = 0, accepted = 0;
    const std::vector<float> query(dim, 0.25f);

    for (int iter = 0; iter < 400; ++iter) {
        std::string bytes = valid;
        const int flips = 1 + static_cast<int>(rng() % 3);
        for (int f = 0; f < flips; ++f) {
            bytes[rng() % bytes.size()] = static_cast<char>(rng() & 0xFF);
        }

        try {
            auto index = try_load(bytes);
            ++accepted;
            if (index->dimension() == dim) {
                static_cast<void>(index->search(VectorView(query.data(), query.size()), 5));
            }
        } catch (const std::exception&) {
            ++rejected;
        }
    }
    EXPECT_GT(rejected, 0u);
    EXPECT_EQ(rejected + accepted, 400u);
}

TEST(HardeningTest, CorruptedHnswSnapshotsNeverCauseOutOfBoundsAccess) {
    constexpr std::size_t dim = 8, n = 200;
    const auto data = random_data(n, dim, 4);
    HNSWIndex index(dim, MetricType::L2, 8, 50, 20);
    for (std::size_t i = 0; i < n; ++i) index.add(i, VectorView(data.data() + i * dim, dim));
    fuzz_snapshot(serialize_to_string(index), dim, 11);
}

TEST(HardeningTest, CorruptedSq8SnapshotsNeverCauseOutOfBoundsAccess) {
    constexpr std::size_t dim = 8, n = 200;
    const auto data = random_data(n, dim, 5);
    ScalarQuantizer q(dim);
    q.train(data);
    HNSWIndexSQ8 index(dim, MetricType::L2, q, 8, 50, 20);
    for (std::size_t i = 0; i < n; ++i) index.add(i, VectorView(data.data() + i * dim, dim));
    fuzz_snapshot(serialize_to_string(index), dim, 12);
}

TEST(HardeningTest, ValidSnapshotsStillRoundTripAfterValidation) {
    constexpr std::size_t dim = 8, n = 100;
    const auto data = random_data(n, dim, 6);
    HNSWIndex index(dim, MetricType::L2, 8, 50, 20);
    for (std::size_t i = 0; i < n; ++i) index.add(i, VectorView(data.data() + i * dim, dim));

    auto loaded = try_load(serialize_to_string(index));
    ASSERT_EQ(loaded->size(), n);
    const VectorView q(data.data() + 7 * dim, dim);
    EXPECT_EQ(loaded->search(q, 1).front().id, 7u);
}

// ------------------------------------------------------------------ HTTP parsing helpers

TEST(HardeningTest, RequestParserAcceptsEmptyAndUnspacedHeaderValues) {
    const std::string raw =
        "POST /api/search HTTP/1.1\r\n"
        "X-Empty:\r\n"
        "X-NoSpace:value\r\n"
        "Host:   example.test  \r\n"
        "Content-Length: 2\r\n"
        "\r\n"
        "{}";
    HttpRequest req;
    ASSERT_NO_THROW(req = HttpRequestParser::parse(raw));
    EXPECT_EQ(req.method, "POST");
    EXPECT_EQ(req.path, "/api/search");
    EXPECT_EQ(req.headers["X-Empty"], "");
    EXPECT_EQ(req.headers["X-NoSpace"], "value");
    EXPECT_EQ(req.headers["Host"], "example.test");

    // A header line that ends exactly at the colon, with no carriage return.
    ASSERT_NO_THROW(HttpRequestParser::parse("GET / HTTP/1.1\nX:\n\n"));
}

TEST(HardeningTest, JsonParsingRejectsNonFiniteComponents) {
    std::vector<float> v;
    EXPECT_TRUE(server::json::parse_vector_payload("{\"vector\":[1.0, 2.5, -3]}", v));
    EXPECT_EQ(v.size(), 3u);

    EXPECT_FALSE(server::json::parse_vector_payload("{\"vector\":[1.0, nan, 3]}", v));
    EXPECT_FALSE(server::json::parse_vector_payload("{\"vector\":[1.0, inf, 3]}", v));
    EXPECT_FALSE(server::json::parse_vector_payload("{\"vector\":[1.0, -infinity]}", v));
    EXPECT_FALSE(server::json::parse_vector_payload("{\"vector\":[1e999]}", v));
}

TEST(HardeningTest, SearchResponseDoesNotWrapLargeIdsIntoConceptLabels) {
    const std::uint64_t wrapped = (std::uint64_t{1} << 32) + 5;  // low 32 bits == concept id 5
    const std::vector<SearchResult> results = {{5, 0.5f}, {wrapped, 0.7f}};
    const std::string json = server::json::build_search_response(results, 2, 1);

    const auto first = json.find("\"id\":5,");
    const auto second = json.find("\"id\":" + std::to_string(wrapped));
    ASSERT_NE(first, std::string::npos);
    ASSERT_NE(second, std::string::npos);
    EXPECT_NE(json.substr(first, second - first).find("\"label\""), std::string::npos);
    EXPECT_EQ(json.substr(second).find("\"label\""), std::string::npos);
}

}  // namespace
