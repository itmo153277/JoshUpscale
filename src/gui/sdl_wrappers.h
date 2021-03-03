// Copyright 2021 Ivanov Viktor

#pragma once

#include <SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>

#include <exception>
#include <memory>
#include <utility>

namespace sdl {

struct SDLException : std::exception {
	SDLException() : std::exception(::SDL_GetError()) {
	}
};

template <typename F, typename... Params>
void callOrThrow(F f, Params &&...params) {
	if (f(std::forward<Params>(params)...) < 0) {
		throw SDLException();
	}
}

template <typename T>
T *allocOrThrow(T *ret) {
	if (!ret) {
		throw SDLException();
	}
	return ret;
}

}  // namespace sdl

namespace smart {

#define DEFINE_SMART_SDL_CLASS(SDL_Class, SDL_Destructor)                    \
	struct SDL_Class                                                         \
	    : std::unique_ptr<::SDL_Class, decltype(&::SDL_Destructor)> {        \
		using unique_ptr =                                                   \
		    std::unique_ptr<::SDL_Class, decltype(&::SDL_Destructor)>;       \
		SDL_Class() : unique_ptr(nullptr, &::SDL_Destructor) {               \
		}                                                                    \
		SDL_Class(std::nullptr_t) : unique_ptr(nullptr, &::SDL_Destructor) { \
		}                                                                    \
		SDL_Class(const SDL_Class &) = delete;                               \
		SDL_Class(SDL_Class &&) noexcept = default;                          \
		SDL_Class(::SDL_Class *p) : unique_ptr(p, &::SDL_Destructor) {       \
		}                                                                    \
	}

DEFINE_SMART_SDL_CLASS(SDL_Window, SDL_DestroyWindow);

DEFINE_SMART_SDL_CLASS(SDL_Renderer, SDL_DestroyRenderer);

DEFINE_SMART_SDL_CLASS(SDL_Texture, SDL_DestroyTexture);

DEFINE_SMART_SDL_CLASS(SDL_mutex, SDL_DestroyMutex);

#undef DEFINE_SMART_SDL_CLASS

struct SDL_AudioDevice {
	SDL_AudioDevice() = delete;
	SDL_AudioDevice(const char *device, const ::SDL_AudioSpec *spec) {
		m_DeviceId = ::SDL_OpenAudioDevice(
		    device, 0, spec, &m_Spec, SDL_AUDIO_ALLOW_ANY_CHANGE);
		if (!m_DeviceId) {
			throw sdl::SDLException();
		}
	}
	SDL_AudioDevice(const SDL_AudioDevice &) = delete;
	SDL_AudioDevice(SDL_AudioDevice &&s) noexcept {
		m_Spec = s.m_Spec;
		m_DeviceId = s.m_DeviceId;
		s.m_DeviceId = 0;
	}
	~SDL_AudioDevice() {
		if (m_DeviceId) {
			::SDL_CloseAudioDevice(m_DeviceId);
		}
	}

	const ::SDL_AudioSpec *getAudioSpec() const {
		return &m_Spec;
	}
	const ::SDL_AudioDeviceID getDeviceId() const {
		return m_DeviceId;
	}

private:
	::SDL_AudioSpec m_Spec = {};
	::SDL_AudioDeviceID m_DeviceId = 0;
};

}  // namespace smart

namespace sdl {

void init();

struct SLockGuard {
	SLockGuard() = delete;
	SLockGuard(const SLockGuard &) = delete;
	SLockGuard(SLockGuard &&) = delete;
	explicit SLockGuard(::SDL_mutex *mutex) : mutex(mutex) {
		sdl::callOrThrow(::SDL_LockMutex, mutex);
	}
	~SLockGuard() {
		sdl::callOrThrow(::SDL_UnlockMutex, mutex);
	}

private:
	::SDL_mutex *mutex;
};

struct STextureLock {
	STextureLock() = delete;
	STextureLock(const STextureLock &) = delete;
	STextureLock(STextureLock &&) = delete;
	STextureLock(::SDL_Texture *texture, void **pixels, int *pitch)
	    : texture(texture) {
		sdl::callOrThrow(::SDL_LockTexture, texture, nullptr, pixels, pitch);
	}
	~STextureLock() {
		::SDL_UnlockTexture(texture);
	}

private:
	::SDL_Texture *texture;
};

}  // namespace sdl
