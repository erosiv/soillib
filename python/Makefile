# soillib/python Makefile
# Compiler Configuration

INCPATH = -I$(HOME)/.local/include
LIBPATH = -L$(HOME)/.local/lib
DIRNAME = soillib
LINKLIB = -ltiff -lnanobind

CC = g++ -std=c++23
CF = -Wfatal-errors -O3

# OS Specific Configuration

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)			# Detect GNU/Linux
endif

ifeq ($(UNAME), Darwin)			# Detect MacOS
INCPATH = -I/opt/homebrew/include
LIBPATH = -L/opt/homebrew/lib
CC = g++-13 -std=c++23
endif

CF += -fvisibility=hidden \
    	-DNDEBUG -DNB_COMPACT_ASSERTIONS \
    	`python3-config --includes` -fPIC \
			-O3 -fno-strict-aliasing \
      -ffunction-sections -fdata-sections

all:
#	Compile the Extension Code
	@$(eval MODULE:= soillib$(shell python3-config --extension-suffix))
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/soillib.cpp $(LINKLIB) -Os -c -o soillib.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/index.cpp $(LINKLIB) -Os -c -o index.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/io.cpp $(LINKLIB) -Os -c -o io.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/util.cpp $(LINKLIB) -Os -c -o util.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/matrix.cpp $(LINKLIB) -Os -c -o matrix.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/node.cpp $(LINKLIB) -Os -c -o node.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/particle.cpp $(LINKLIB) -Os -c -o particle.o
	@echo "Linking..."
	@$(CC) -shared -Wl,-s -Wl,--gc-sections -ltiff \
			soillib.o \
			index.o \
			io.o \
			util.o \
			matrix.o \
			node.o \
			particle.o \
		$(HOME)/.local/lib/libnanobind.o -o ${MODULE}
	@mv ${MODULE} $(HOME)/.local/lib/python3.12/site-packages/${MODULE}
	@rm ./*.o