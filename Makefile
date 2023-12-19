# soillib
# Author: Nicholas McDonald
# Version 1.0
# Tested on GNU/Linux and MacOS

# Install Location

LIBPATH = $(HOME)/.local/lib
INCPATH = $(HOME)/.local/include
DIRNAME = soillib

# Compiler Settings

CC = g++ -std=c++20
CF = -Wfatal-errors -O2

# OS Specific Configuration

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)			# Detect GNU/Linux
endif

ifeq ($(UNAME), Darwin)			# Detect MacOS

INCPATH = /opt/homebrew/include
LIBPATH = /opt/homebrew/lib

CC = g++-13 -std=c++20

endif

##########################
#  	soillib installer  	 #
##########################

.PHONY: all
all:
	@echo "soillib: copying header files...";
	@# Prepare Directories
	@if [ ! -d $(INCPATH) ]; then mkdir $(INCPATH); fi;
	@if [ -d $(INCPATH)/$(DIRNAME) ]; then rm -rf $(INCPATH)/$(DIRNAME); fi;
	@mkdir $(INCPATH)/$(DIRNAME)
	@# Copy Files
	@cp -r soillib/* $(INCPATH)/$(DIRNAME)
	@echo "soillib: done";
