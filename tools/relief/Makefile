# soillib
# Author: Nicholas McDonald
# Version 1.0
# Tested on GNU/Linux and MacOS

# Install Location

LIBPATH = $(HOME)/.local/lib
INCPATH = $(HOME)/.local/include
DIRNAME = soillib

# Compiler Settings

CC = g++-10 -std=c++20
CF = -Wfatal-errors -O2

# OS Specific Configuration

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)			# Detect GNU/Linux
endif

ifeq ($(UNAME), Darwin)			# Detect MacOS

INCPATH = /opt/homebrew/include
LIBPATH = /opt/homebrew/lib

CC = g++-13 -std=c++20

LINK = -lpng

endif

all: main.cpp
			$(CC) $(CF) -L$(LIBPATH) -I$(INCPATH) $(LINK) main.cpp -o main