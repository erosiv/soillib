# Install Location

INCPATH = $(HOME)/.local/include
LIBPATH = $(HOME)/.local/lib
DIRNAME = soillib

# Compiler Settings

CC = g++ -std=c++23
CF = -Wfatal-errors -O2

# OS Specific Configuration

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)			# Detect GNU/Linux
endif

ifeq ($(UNAME), Darwin)			# Detect MacOS

INCPATH = /opt/homebrew/include
LIBPATH = /opt/homebrew/lib
CC = g++-13 -std=c++23

endif

##########################
#  	soillib installer  	 #
##########################

BUILDDIR 	= build
LIBDIST = libsoil

#WHLDIST 	= soillib
#VERSION 	= 1.0.0
#WHLTAG 		= py3-none-manylinux_2_17_x86_64

GCCBIN = /usr/bin/gcc-13
CC = nvcc -ccbin $(GCCBIN) -std c++20
INC_CUDA = -I/usr/local/cuda-12/include
LINKLIB = #-ltiff -lnanobind

.PHONY: all
all:
	@echo "soillib: copying header files...";
	@# Prepare Directories
	@if [ ! -d $(INCPATH) ]; then mkdir $(INCPATH); fi;
	@if [ -d $(INCPATH)/$(DIRNAME) ]; then rm -rf $(INCPATH)/$(DIRNAME); fi;
	@mkdir $(INCPATH)/$(DIRNAME)
	@# Copy Files
	@cp -r ./* $(INCPATH)/$(DIRNAME)
	@echo "Building Library: ${LIBDIST}"
	@if [ ! -d $(BUILDDIR) ]; then mkdir $(BUILDDIR); fi;
# Build CUDA Files
	@$(CC) --shared -diag-suppress 20012 -Xcompiler -fPIC -I$(INCPATH) -L$(LIBPATH) $(INCCUDA) node/algorithm/flow.cu $(LINKLIB) -O3 -c -o $(BUILDDIR)/$(LIBDIST).so
	@echo "Installing Library ($(LIBDIST))..."
	@mv $(BUILDDIR)/$(LIBDIST).so $(LIBPATH)/$(LIBDIST).so
	@rmdir $(BUILDDIR)

.PHONY: lint
lint:
	@clang-format -i ./soillib.hpp
	@clang-format -i ./core/*.hpp
	@clang-format -i ./index/*.hpp
	@clang-format -i ./io/*.hpp
	@clang-format -i ./node/*.hpp
	@clang-format -i ./node/algorithm/*.hpp
	@clang-format -i ./particle/*.hpp
	@clang-format -i ./util/*.hpp