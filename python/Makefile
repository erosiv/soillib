# soillib/python Makefile
# Compiler Configuration

INCPATH = -I$(HOME)/.local/include
LIBPATH = -L$(HOME)/.local/lib
DIRNAME = soillib
LINKLIB = -lsoil -ltiff -lnanobind 

CC = g++ -std=c++23
CF = -Wfatal-errors -O3 -Os

# OS Specific Configuration

UNAME := $(shell uname)

# ifeq ($(UNAME), Linux)			# Detect GNU/Linux
# endif
# 
# ifeq ($(UNAME), Darwin)			# Detect MacOS
# INCPATH = -I/opt/homebrew/include
# LIBPATH = -L/opt/homebrew/lib
# CC = g++-13 -std=c++23
# endif

# Wheel File / Distributable Configuration

BUILDDIR 	= wheel
WHLDIST 	= soillib
VERSION 	= 1.0.0
WHLTAG 		= py3-none-manylinux_2_17_x86_64

#
# Binary Compilation
#

CF += -fvisibility=hidden \
    	-DNDEBUG -DNB_COMPACT_ASSERTIONS \
    	`python3-config --includes` \
			-fno-strict-aliasing \
      -ffunction-sections -fdata-sections

.PHONY: build
build:
#	Compile the Extension Code
	@$(eval MODULE:= soillib$(shell python3-config --extension-suffix))
	@echo "  Building Python Module: ${MODULE}"
	@echo "  Compiling Binding Object Files..."
	@if [ ! -d $(BUILDDIR) ]; then mkdir $(BUILDDIR); fi;
	@$(CC) $(CF) $(INCPATH) -fPIC	-c source/soillib.cpp 	-o $(BUILDDIR)/soillib.o
	@$(CC) $(CF) $(INCPATH) -fPIC	-c source/index.cpp 	 	-o $(BUILDDIR)/index.o
	@$(CC) $(CF) $(INCPATH) -fPIC	-c source/io.cpp 				-o $(BUILDDIR)/io.o
	@$(CC) $(CF) $(INCPATH) -fPIC	-c source/util.cpp 			-o $(BUILDDIR)/util.o
	@$(CC) $(CF) $(INCPATH) -fPIC	-c source/matrix.cpp 		-o $(BUILDDIR)/matrix.o
	@$(CC) $(CF) $(INCPATH) -fPIC	-c source/model.cpp 	  -o $(BUILDDIR)/model.o
	@$(CC) $(CF) $(INCPATH) -fPIC	-c source/node.cpp 	 		-o $(BUILDDIR)/node.o
	@$(CC) $(CF) $(INCPATH) -fPIC	-c source/particle.cpp 	-o $(BUILDDIR)/particle.o
	@echo "  Linking Object Files..."
	@$(CC) -shared -o $(BUILDDIR)/${MODULE} \
		$(LIBPATH) \
		$(BUILDDIR)/soillib.o \
		$(BUILDDIR)/index.o \
		$(BUILDDIR)/io.o \
		$(BUILDDIR)/util.o \
		$(BUILDDIR)/matrix.o \
		$(BUILDDIR)/model.o \
		$(BUILDDIR)/node.o \
		$(BUILDDIR)/particle.o \
		$(HOME)/.local/lib/libnanobind.o \
		-lnanobind \
		-lsoil \
		-ltiff \
		-L/usr/local/cuda-12/lib64 -lcudart -lcuda
	@rm $(BUILDDIR)/*.o

#
# Wheel File Builder
#

.PHONY: wheel
wheel:
#	Packaging Parameters
	@echo "  Building Wheel File..."
	@$(eval WHEEL:= ${WHLDIST}-${VERSION}-${WHLTAG}.whl)
	@$(eval INFODIR:= ${WHLDIST}-${VERSION}.dist-info)
	@$(eval DATADIR:= ${WHLDIST}-${VERSION}.data)
	@if [ ! -d $(BUILDDIR) ]; then mkdir $(BUILDDIR); fi;
	@if [ ! -d $(BUILDDIR)/$(INFODIR) ]; then mkdir $(BUILDDIR)/$(INFODIR); fi;
	@if [ ! -d $(BUILDDIR)/$(DATADIR) ]; then mkdir $(BUILDDIR)/$(DATADIR); fi;
# Copy Files into Directory
	@cp METADATA $(BUILDDIR)/$(INFODIR)/METADATA
	@cp WHEEL $(BUILDDIR)/$(INFODIR)/WHEEL
# RECORD File
	@rm ${BUILDDIR}/${INFODIR}/RECORD 	&> /dev/null ||:
	@touch ${BUILDDIR}/${INFODIR}/RECORD
	@$(eval FILES:= \
		${WHLDIST}-${VERSION}.dist-info/WHEEL \
		${WHLDIST}-${VERSION}.dist-info/METADATA \
		${MODULE})
	@cd ${BUILDDIR}; $(foreach FILE,$(FILES), \
		echo -n "${FILE},sha256=" >> ${WHLDIST}-${VERSION}.dist-info/RECORD; \
		cat ${FILE} | sha256sum >> ${WHLDIST}-${VERSION}.dist-info/RECORD;)
	@echo "  Compressing..."
	@cd ${BUILDDIR}; zip -r ../${WHEEL} .;

#
# Wheel File Installation
#

.PHONY: install
install:
	@$(eval WHEEL:= ${WHLDIST}-${VERSION}-${WHLTAG}.whl)
	@echo "Installing Wheel (${WHEEL})..."
	@pip install ./$(WHEEL) --force-reinstall
	@rm ./$(WHEEL)

.PHONY: all
all: build wheel install