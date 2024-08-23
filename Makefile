# soillib
# Version: 1.0
# Author: Nicholas McDonald

#
# Build / Installation Rules
#

.PHONY: source
source:
	@echo "soillib: building and installing from source..."
	@cd source; $(MAKE) --no-print-directory all
	@echo "soillib: done"

.PHONY: python
python:
	@echo "soillib: building python module..."
	@cd python; $(MAKE) --no-print-directory all
	@echo "soillib: done"

.PHONY: test
test:
	@echo "soillib: running test scripts..."
	@cd test; $(MAKE) --no-print-directory all
	@echo "soillib: done"

.PHONY: all
all: source python