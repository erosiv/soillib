# soillib
# Version: 1.0
# Author: Nicholas McDonald

#
# Build / Installation Rules
#

.PHONY: python
python:
	@echo "soillib: building python module..."
	@cd python; $(MAKE) --no-print-directory all
	@echo "soillib: done"

.PHONY: all
all: #source python
	rm -rf build
	cmake -S . -B build
	cmake --build build
	cmake --install build

.PHONY: test
test:
	@echo "soillib: running test scripts..."
	@cd test; $(MAKE) --no-print-directory all
	@echo "soillib: done"

.PHONY: lint
lint:
	@echo "soillib: running clang-format..."
	@cd source; $(MAKE) --no-print-directory lint
	@echo "soillib: done"