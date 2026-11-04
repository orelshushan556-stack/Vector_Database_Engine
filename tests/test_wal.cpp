#include <gtest/gtest.h>
#include "engine/storage/wal_manager.hpp"
#include <filesystem>

namespace fs = std::filesystem;

class WalManagerBasicTest : public ::testing::Test {
protected:
    const std::string test_wal_path = "test_run.wal";

    void SetUp() override {
        if (fs::exists(test_wal_path)) {
            fs::remove(test_wal_path);
        }
    }

    void TearDown() override {
        if (fs::exists(test_wal_path)) {
            fs::remove(test_wal_path);
        }
    }
};

TEST_F(WalManagerBasicTest, CreatesFileAndOpensSuccessfully) {
    {
        engine::storage::WalManager wal(test_wal_path);
        EXPECT_TRUE(wal.is_open());
    }
    // לאחר יציאה מהבלוק והפעלת ה-Destructor, הקובץ חייב להתקיים בדיסק
    EXPECT_TRUE(fs::exists(test_wal_path));
}

TEST_F(WalManagerBasicTest, SyncToDiskExecutesWithoutErrors) {
    engine::storage::WalManager wal(test_wal_path);
    EXPECT_TRUE(wal.is_open());
    
    // קריאה ישירה לפקודת סנכרון החומרה של מערכת ההפעלה
    EXPECT_NO_THROW(wal.sync());
}

TEST_F(WalManagerBasicTest, ThrowsOnInvalidPath) {
    // נתיב לתיקייה שאינה קיימת כלל
    EXPECT_THROW({
        engine::storage::WalManager wal("non_existing_dir_xyz/test.wal");
    }, std::runtime_error);
}
