# Compiler and compiler flags
CC = gcc
CFLAGS = -O3 -Wno-unused-result
LDFLAGS = -lm

# Target executable names
TARGETS = train recognize

# Default target
all: $(TARGETS)

train: train.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

recognize: recognize.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f *.o $(TARGETS)

# Phony targets
.PHONY: all clean

