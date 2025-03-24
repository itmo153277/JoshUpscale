// Copyright 2025 Ivanov Viktor

#pragma once

#include <exception>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace JoshUpscale {

namespace core {

struct ExceptionWithIdBase {
	virtual const std::type_info &type_info() const noexcept = 0;
};

template <typename T>
struct ExceptionWithId : T, ExceptionWithIdBase {
	explicit ExceptionWithId(T &&base) : T(std::move(base)) {
	}

	const std::type_info &type_info() const noexcept override {
		return typeid(T);
	}
};

template <typename T>
ExceptionWithId<std::decay_t<T>> make_exception_with_id(T &&exception) {
	return ExceptionWithId<std::decay_t<T>>{std::forward<T>(exception)};
}

template <typename T>
[[noreturn]] void throw_with_nested_id(T &&exception) {
	std::throw_with_nested(make_exception_with_id(std::forward<T>(exception)));
}

void printException(
    std::ostream &os, const std::exception &e, const std::type_info &info);

void printException(std::ostream &os, const std::exception &e);

void printException(std::ostream &os);

}  // namespace core

}  // namespace JoshUpscale
