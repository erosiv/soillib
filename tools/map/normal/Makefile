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

# OS Specific Configuration

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)			# Detect GNU/Linux

INCPATH = -I$(HOME)/.local/include
LIBPATH = -L$(HOME)/.local/lib

LINK = -lpng -ltiff

endif

ifeq ($(UNAME), Darwin)			# Detect MacOS

INCPATH = -I/opt/homebrew/include
LIBPATH = -L/opt/homebrew/lib

CC = g++-13 -std=c++20

LINK = -lpng -ltiff

endif

all: main.cpp
			$(CC) main.cpp $(CF) $(LIBPATH) $(INCPATH) $(LINK) -o main
