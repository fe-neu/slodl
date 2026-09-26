#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <thread>

#include "slodl/autograd/grad_mode.hpp"

TEST_CASE("recording is on by default", "[autograd]") {
    CHECK(is_grad_enabled());
}

TEST_CASE("set_grad_enabled switches recording", "[autograd]") {
    set_grad_enabled(false);
    CHECK_FALSE(is_grad_enabled());

    set_grad_enabled(true);
    CHECK(is_grad_enabled());
}

TEST_CASE("NoGradGuard disables recording for its lifetime", "[autograd]") {
    {
        NoGradGuard guard;
        CHECK_FALSE(is_grad_enabled());
    }

    CHECK(is_grad_enabled());
}

TEST_CASE("NoGradGuards nest", "[autograd]") {
    NoGradGuard outer;
    {
        NoGradGuard inner;
        CHECK_FALSE(is_grad_enabled());
    }

    // The inner guard restores what it found, not the default.
    CHECK_FALSE(is_grad_enabled());
}

TEST_CASE("NoGradGuard restores while unwinding", "[autograd]") {
    try {
        NoGradGuard guard;
        throw std::runtime_error("boom");
    }
    catch (const std::runtime_error&) {
    }

    CHECK(is_grad_enabled());
}

TEST_CASE("grad mode is per thread", "[autograd]") {
    NoGradGuard guard;
    bool enabled_in_thread = false;

    std::thread worker([&enabled_in_thread]() {
        enabled_in_thread = is_grad_enabled();
    });
    worker.join();

    CHECK_FALSE(is_grad_enabled());
    CHECK(enabled_in_thread);
}
