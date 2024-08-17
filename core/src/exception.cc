// Copyright 2022 Ivanov Viktor

#include "JoshUpscale/core/exception.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

#ifdef __GNUG__
#include <cxxabi.h>

#include <cstdlib>
#include <memory>
#elif defined(_MSC_VER)
#include <cstddef>
#endif

namespace JoshUpscale {

namespace core {

namespace {

std::string demangle(const char *name) {
#if defined(__GNUG__)
	int status = EXIT_FAILURE;
	std::unique_ptr<char, void (*)(void *)> res{
	    abi::__cxa_demangle(name, NULL, NULL, &status), std::free};

	return (status == EXIT_SUCCESS) ? res.get() : name;
#elif defined(_MSC_VER)
	static const char *prefixes[] = {
	    "class ", "struct ", "union ", "enum ", ""};
	static const std::size_t prefixLens[] = {6, 7, 6, 5, 0};
	const char **prefixPtr = prefixes;
	const std::size_t *prefixLenPtr = prefixLens;
	for (; **prefixPtr != 0; prefixPtr++, prefixLenPtr++) {
		if (std::strncmp(name, *prefixPtr, *prefixLenPtr) == 0) {
			return name + *prefixLenPtr;
		}
	}
	return name;
#else
	return name;
#endif
}

}  // namespace

void printException(
    std::ostream &os, const std::exception &e, const std::type_info &info) {
	os << demangle(info.name()) << ": " << e.what();
	try {
		std::rethrow_if_nested(e);
	} catch (...) {
		os << "\n  ";
		printException(os);
	}
}

void printException(std::ostream &os, const std::exception &e) {
	const auto *typedExc = dynamic_cast<const ExceptionWithIdBase *>(&e);
	if (typedExc == nullptr) {
		printException(os, e, typeid(e));
	} else {
		printException(os, e, typedExc->type_info());
	}
}

void printException(std::ostream &os) {
	try {
		throw;
	} catch (std::exception &e) {
		printException(os, e);
	} catch (...) {
		os << "Unknown error";
	}
}

std::string getExceptionString() {
	std::ostringstream ss;
	printException(ss);
	return ss.str();
}

}  // namespace core

}  // namespace JoshUpscale
