// Copyright 2025 Viktor Ivanov

#pragma once

#include <utility>

namespace JoshUpscale {

namespace obs {

namespace detail {

template <typename T>
struct Defer {
	T m_DeferFn;
	explicit Defer(T &&fn) noexcept : m_DeferFn(std::move(fn)) {
	}
	Defer(const Defer &) = delete;
	Defer(Defer &&) noexcept = delete;
	~Defer() {
		m_DeferFn();
	}
};

struct DeferOp {};

template <typename T>
auto operator+([[maybe_unused]] DeferOp op, T &&fn) noexcept {
	return Defer{std::forward<T>(fn)};
}

}  // namespace detail

// NOLINTBEGIN
#define DEFER_OP_NAME_(LINE) _defer##LINE
#define DEFER_OP_NAME(LINE) DEFER_OP_NAME_(LINE)
#define defer                      \
	auto DEFER_OP_NAME(__LINE__) = \
	    ::JoshUpscale::obs::detail::DeferOp{} + [&]() noexcept -> void
// NOLINTEND

}  // namespace obs

}  // namespace JoshUpscale
