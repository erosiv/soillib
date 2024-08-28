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
	@cp -r ./* $(INCPATH)/$(DIRNAME)

.PHONY: lint
lint:
	@echo "soillib: linting with clang-format..."
	@clang-format -i ./soillib.hpp
	@clang-format -i ./core/*.hpp
	@clang-format -i ./index/*.hpp
	@clang-format -i ./io/*.hpp
	@clang-format -i ./node/*.hpp
	@clang-format -i ./node/algorithm/*.hpp
	@clang-format -i ./particle/*.hpp
	@clang-format -i ./util/*.hpp