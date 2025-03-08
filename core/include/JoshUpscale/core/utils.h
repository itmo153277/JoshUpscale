// Copyright 2025 Ivanov Viktor

#pragma once

#include <algorithm>
#include <cassert>

namespace JoshUpscale {

namespace core {

template <typename... T>
struct MultiClassStorage {
	char *getPtr() const {
		return const_cast<char *>(m_Storage);
	}

private:
	alignas(T...) char m_Storage[std::max({sizeof(T)...})];
};

[[noreturn]] inline void unreachable() {
#ifdef NDEBUG
#ifdef _MSC_VER
	__assume(false);
#else
	__builtin_unreachable();
#endif
#else
	assert(false);
#endif
}

}  // namespace core

}  // namespace JoshUpscale
