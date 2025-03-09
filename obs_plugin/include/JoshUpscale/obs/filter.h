// Copyright 2025 Viktor Ivanov

#pragma once

extern "C" {
#include <obs-module.h>
}

#include <JoshUpscale/core.h>

#include <memory>
#include <utility>

namespace JoshUpscale {

namespace obs {

namespace detail {

struct OBSDeleter {
	void operator()(void *data) {
		::bfree(data);
	}
};

}  // namespace detail

template <class T>
struct OBSPtr : std::unique_ptr<T, detail::OBSDeleter> {
	using unique_ptr = std::unique_ptr<T, detail::OBSDeleter>;

	explicit OBSPtr(T *val) : unique_ptr(val) {
	}
};

template <typename T>
struct Defer {
	T m_DeferFn;
	explicit Defer(T &&fn) : m_DeferFn(std::move(fn)) {
	}
	Defer(const Defer &) = delete;
	Defer(Defer &&) noexcept = delete;
	~Defer() {
		m_DeferFn();
	}
};

struct DeferOp {};

template <typename T>
Defer<T> operator*([[maybe_unused]] DeferOp op, T &&fn) {
	return Defer{std::forward<T>(fn)};
}

#define DEFER_OP_NAME_(LINE) _defer##LINE
#define DEFER_OP_NAME(LINE) DEFER_OP_NAME_(LINE)
#define defer auto DEFER_OP_NAME(__LINE__) = DeferOp{} *[&]()

struct JoshUpscaleFilter {
	static ::obs_source_info *getSourceInfo();

private:
	JoshUpscaleFilter(::obs_data_t *settings, ::obs_source_t *source);
	~JoshUpscaleFilter();

	static const char *getName(void *typeData) noexcept;

	static void *create(
	    ::obs_data_t *settings, ::obs_source_t *source) noexcept;

	static void destroy(void *data) noexcept;

	void render(::gs_effect_t *effect) noexcept;

	template <auto Ptr>
	struct Callback {
		static consteval decltype(Ptr) getPtr() noexcept {
			return Ptr;
		}
	};

	template <typename R, typename... T,
	    R (JoshUpscaleFilter::*Ptr)(T...) noexcept>
	struct Callback<Ptr> {
		static consteval R (*getPtr() noexcept)(void *, T...) noexcept {
			return [](void *self, T... params) noexcept -> R {
				return (reinterpret_cast<JoshUpscaleFilter *>(self)->*Ptr)(
				    params...);
			};
		}
	};

	::obs_source_t *m_Source;
};

}  // namespace obs

}  // namespace JoshUpscale
