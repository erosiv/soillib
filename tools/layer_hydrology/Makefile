# soillib
# Author: Nicholas McDonald
# Version 1.0
# Tested on GNU/Linux and MacOS

# Install Location

INCPATH = -I$(HOME)/.local/include
LIBPATH = -L$(HOME)/.local/lib
DIRNAME = soillib

# Compiler Settings

CC = g++-10 -std=c++20
CF = -Wfatal-errors -O2

TINYLINK = -lpthread -lSDL2 -lSDL2_image -lSDL2_mixer -lSDL2_ttf -lGLEW -lboost_system -lboost_filesystem

# OS Specific Configuration

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)			# Detect GNU/Linux

INCPATH = -I$(HOME)/.local/include
LIBPATH = -L$(HOME)/.local/lib

LINK = -lpng -ltiff

endif
## Detect MacOS
ifeq ($(UNAME), Darwin)
TINYLINK += -framework OpenGL -DTINYENGINE_OS_MAC
LF += -I/opt/homebrew/include -L/opt/homebrew/lib
CC = g++-13 -std=c++20
endif

all: main.cpp
			$(CC) main.cpp -o main $(CF) $(LF) -lTinyEngine $(TINYLINK) -ltiff -lpng
