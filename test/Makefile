# soillib/test

TESTS = ./test_buffer.py ./test_index.py ./test_node.py

.PHONY: all
all:
	@$(eval FILE:= ./test_buffer.py)
	@$(foreach FILE,$(TESTS), \
		python3 $(FILE) \
			> /dev/null \
			2> /dev/null \
			&& echo "Test (${FILE}): SUCCESS" \
			|| echo "Test (${FILE}): FAILURE" \
	;)
