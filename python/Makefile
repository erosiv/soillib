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

# Wheel File / Distributable Configuration

BUILDDIR = "wheel"

#
# Binary Compilation
#

all:
#	Compile the Extension Code
	@$(eval MODULE:= soillib$(shell python3-config --extension-suffix))
	@echo "Building Module: ${MODULE}"
	@if [ ! -d $(BUILDDIR) ]; then mkdir $(BUILDDIR); fi;
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/soillib.cpp 	$(LINKLIB) -Os -c -o $(BUILDDIR)/soillib.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/index.cpp 		$(LINKLIB) -Os -c -o $(BUILDDIR)/index.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/io.cpp 				$(LINKLIB) -Os -c -o $(BUILDDIR)/io.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/util.cpp 			$(LINKLIB) -Os -c -o $(BUILDDIR)/util.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/matrix.cpp 		$(LINKLIB) -Os -c -o $(BUILDDIR)/matrix.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/model.cpp 		$(LINKLIB) -Os -c -o $(BUILDDIR)/model.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/node.cpp 			$(LINKLIB) -Os -c -o $(BUILDDIR)/node.o
	@$(CC) $(CF) $(INCPATH) $(LIBPATH) source/particle.cpp 	$(LINKLIB) -Os -c -o $(BUILDDIR)/particle.o
	@echo "Linking..."
	@$(CC) -shared -Wl,-s -Wl,--gc-sections -ltiff \
			$(BUILDDIR)/soillib.o \
			$(BUILDDIR)/index.o \
			$(BUILDDIR)/io.o \
			$(BUILDDIR)/util.o \
			$(BUILDDIR)/matrix.o \
			$(BUILDDIR)/model.o \
			$(BUILDDIR)/node.o \
			$(BUILDDIR)/particle.o \
		$(HOME)/.local/lib/libnanobind.o -o $(BUILDDIR)/${MODULE}
	@rm $(BUILDDIR)/*.o
#
# Wheel File Builder
#

.PHONY: wheel
wheel:
	@if [ ! -d $(BUILDDIR) ]; then mkdir $(BUILDDIR); fi;
#	Packaging Parameters
	@$(eval WHLDIST:= soillib)
	@$(eval VERSION:= 1.0.0)
	@$(eval WHLTAG:= py3-none-linux_x86_64)
	@$(eval WHEEL:= ${WHLDIST}-${VERSION}-${WHLTAG}.whl)
# Construct Directory Hierarchy
	@$(eval INFODIR:= ${WHLDIST}-${VERSION}.dist-info)
	@$(eval DATADIR:= ${WHLDIST}-${VERSION}.data)
	@if [ ! -d $(BUILDDIR)/$(INFODIR) ]; then mkdir $(BUILDDIR)/$(INFODIR); fi;
	@if [ ! -d $(BUILDDIR)/$(DATADIR) ]; then mkdir $(BUILDDIR)/$(DATADIR); fi;
# Add dist-info
# WHEEL File
	@rm ${BUILDDIR}/${INFODIR}/WHEEL 		&> /dev/null ||:
	@touch ${BUILDDIR}/${INFODIR}/WHEEL
	@echo "Wheel-Version: 1.0" 					>> ${BUILDDIR}/${INFODIR}/WHEEL
	@echo "Root-Is-Purelib: true" 			>> ${BUILDDIR}/${INFODIR}/WHEEL
	@echo "Tag: ${WHLTAG}" 							>> ${BUILDDIR}/${INFODIR}/WHEEL
	@echo "Build: ${VERSION}"						>> ${BUILDDIR}/${INFODIR}/WHEEL
# METADATA File
	@rm ${BUILDDIR}/${INFODIR}/METADATA &> /dev/null ||:
	@touch ${BUILDDIR}/${INFODIR}/METADATA
	@echo "Metadata-Version: 1.1"																				>> ${BUILDDIR}/${INFODIR}/METADATA
	@echo "Name: ${WHLDIST}"																						>> ${BUILDDIR}/${INFODIR}/METADATA	
	@echo "Version: ${VERSION}"																					>> ${BUILDDIR}/${INFODIR}/METADATA	
	@echo "Platform: Unix"																							>> ${BUILDDIR}/${INFODIR}/METADATA
	@echo "Summary: ..."																								>> ${BUILDDIR}/${INFODIR}/METADATA
	@echo "Author-email: "N. McDonald" <nicholasmcdonald40@gmail.com>" 	>> ${BUILDDIR}/${INFODIR}/METADATA
	@echo "License: License" 																						>> ${BUILDDIR}/${INFODIR}/METADATA
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

# Zip Wheel File!
	@cd ${BUILDDIR}; zip -r ../${WHEEL} .;