#include <gtest/gtest.h>
#include "engine/quantization/scalar_quantizer.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>

namespace engine {
namespace {

TEST(ScalarQuantizerTest, ThrowsWhenEncodingBeforeTraining) {
    ScalarQuantizer quantizer(4);
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_FALSE(quantizer.is_trained());
    EXPECT_THROW(quantizer.encode(input), std::runtime_error);

    std::vector<uint8_t> encoded(4);
    EXPECT_THROW(quantizer.encode(input, encoded), std::runtime_error);
    EXPECT_THROW(quantizer.decode(encoded), std::runtime_error);
}

TEST(ScalarQuantizerTest, ThrowsOnInvalidTrainingData) {
    ScalarQuantizer quantizer(4);
    std::vector<float> empty_data;
    EXPECT_THROW(quantizer.train(empty_data), std::invalid_argument);

    std::vector<float> misaligned_data = {1.0f, 2.0f, 3.0f};
    EXPECT_THROW(quantizer.train(misaligned_data), std::invalid_argument);

    ScalarQuantizer zero_dim(0);
    std::vector<float> valid_data = {1.0f, 2.0f};
    EXPECT_THROW(zero_dim.train(valid_data), std::invalid_argument);
}

TEST(ScalarQuantizerTest, ThrowsOnDimensionMismatch) {
    ScalarQuantizer quantizer(4);
    std::vector<float> training_data = {
        0.0f, 10.0f, 20.0f, 30.0f,
        10.0f, 20.0f, 30.0f, 40.0f
    };
    quantizer.train(training_data);

    std::vector<float> wrong_size_input = {1.0f, 2.0f, 3.0f};
    EXPECT_THROW(quantizer.encode(wrong_size_input), std::invalid_argument);

    std::vector<uint8_t> valid_encoded = {0, 128, 200, 255};
    std::vector<float> wrong_size_dst(3);
    EXPECT_THROW(quantizer.decode(valid_encoded, wrong_size_dst), std::invalid_argument);
}

TEST(ScalarQuantizerTest, HandlesZeroRangeAndConstantDimensions) {
    constexpr std::size_t dim = 4;
    ScalarQuantizer quantizer(dim);

    std::vector<float> training_data = {
        5.0f, 0.0f, 10.0f, 2.0f,
        5.0f, 10.0f, 20.0f, 2.0f,
        5.0f, 20.0f, 30.0f, 2.0f
    };
    quantizer.train(training_data);

    EXPECT_TRUE(quantizer.is_trained());
    EXPECT_FLOAT_EQ(quantizer.scale_values()[0], 1.0f);
    EXPECT_FLOAT_EQ(quantizer.scale_values()[3], 1.0f);

    std::vector<float> input = {5.0f, 10.0f, 20.0f, 2.0f};
    auto encoded = quantizer.encode(input);

    EXPECT_EQ(encoded[0], 0);
    EXPECT_EQ(encoded[3], 0);

    auto decoded = quantizer.decode(encoded);
    EXPECT_FLOAT_EQ(decoded[0], 5.0f);
    EXPECT_FLOAT_EQ(decoded[3], 2.0f);
}

TEST(ScalarQuantizerTest, RoundTripPrecisionWithAVXAndTail) {
    constexpr std::size_t dim = 11;
    ScalarQuantizer quantizer(dim);

    std::vector<float> training_data = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f
    };
    quantizer.train(training_data);

    std::vector<float> test_vec = {
        5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
    };

    auto encoded = quantizer.encode(test_vec);
    ASSERT_EQ(encoded.size(), dim);

    for (std::size_t i = 0; i < dim; ++i) {
        EXPECT_NEAR(encoded[i], 128, 1);
    }

    auto decoded = quantizer.decode(encoded);
    ASSERT_EQ(decoded.size(), dim);

    for (std::size_t i = 0; i < dim; ++i) {
        float expected_tolerance = quantizer.scale_values()[i];
        EXPECT_NEAR(decoded[i], test_vec[i], expected_tolerance);
    }
}

TEST(ScalarQuantizerTest, ClampingBeyondTrainedRange) {
    constexpr std::size_t dim = 2;
    ScalarQuantizer quantizer(dim);

    std::vector<float> training_data = {
        0.0f, 0.0f,
        100.0f, 100.0f
    };
    quantizer.train(training_data);

    std::vector<float> out_of_bounds = {-50.0f, 200.0f};
    auto encoded = quantizer.encode(out_of_bounds);

    EXPECT_EQ(encoded[0], 0);
    EXPECT_EQ(encoded[1], 255);

    auto decoded = quantizer.decode(encoded);
    EXPECT_FLOAT_EQ(decoded[0], 0.0f);
    EXPECT_FLOAT_EQ(decoded[1], 100.0f);
}

}
}