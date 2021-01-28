#include "sdl_wrappers.h"

void sdl::init() {
	sdl::callOrThrow(
	    ::SDL_Init, SDL_INIT_AUDIO | SDL_INIT_VIDEO | SDL_INIT_EVENTS);
	std::atexit(::SDL_Quit);
}
