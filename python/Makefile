# soillib/python Makefile
# Compiler Configuration

INCPATH = -I$(HOME)/.local/include
LIBPATH = -L$(HOME)/.local/lib
DIRNAME = soillib
LINKLIB = 

CC = g++ -std=c++20
CF = -Wfatal-errors -O3

# OS Specific Configuration

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)			# Detect GNU/Linux
endif

ifeq ($(UNAME), Darwin)			# Detect MacOS
INCPATH = -I/opt/homebrew/include
LIBPATH = -L/opt/homebrew/lib
CC = g++-13 -std=c++20
endif

all:
	@$(eval MODULE:= soillib$(shell python3-config --extension-suffix))
	@$(CC) $(CF) $(LF) $(INCPATH) $(LIBPATH) \
		source/soillib.cpp \
		source/util.cpp \
	$(LINKLIB) -shared -fPIC $(shell python3 -m pybind11 --includes) -o ${MODULE}
	@echo "soillib: installing python module..."
	@mv ${MODULE} $(HOME)/.local/lib/python3.12/site-packages/${MODULE}