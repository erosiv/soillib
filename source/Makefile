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

# Note: This means we compile a pure object file, no linking,
# which has relocatable code.
CF = -diag-suppress 20012 -diag-suppress 445 -Xcompiler -fPIC -c -O3 -Xcompiler #-fno-char8_t

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

	@echo "Compiling Source Code..."
	@$(CC) $(CF) -I$(INCPATH) -L$(LIBPATH) $(INCCUDA) core/buffer.cu 	-o $(BUILDDIR)/buffer.o
	@$(CC) $(CF) -I$(INCPATH) -L$(LIBPATH) $(INCCUDA) node/flow.cu 		-o $(BUILDDIR)/flow.o
	@$(CC) $(CF) -I$(INCPATH) -L$(LIBPATH) $(INCCUDA) node/common.cu 	-o $(BUILDDIR)/common.o
	@nvcc -dlink -Xcompiler -fPIC -o $(BUILDDIR)/$(LIBDIST).o \
		$(BUILDDIR)/buffer.o \
		$(BUILDDIR)/flow.o \
		$(BUILDDIR)/common.o
	@nvcc -lib -o $(BUILDDIR)/$(LIBDIST).a $(BUILDDIR)/$(LIBDIST).o \
		$(BUILDDIR)/buffer.o \
		$(BUILDDIR)/flow.o \
		$(BUILDDIR)/common.o

	@echo "Installing Library ($(LIBDIST))..."
	@mv $(BUILDDIR)/$(LIBDIST).a $(LIBPATH)/$(LIBDIST).a
	@mv $(BUILDDIR)/$(LIBDIST).o $(LIBPATH)/$(LIBDIST).o
	@rm $(BUILDDIR)/*.o
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