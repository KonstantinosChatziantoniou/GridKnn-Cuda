CC = gcc
CFLAGS = -pg -Ofast -std=c99
LIBS = -lm -Wall -Wextra
OBJDIR = ./bin
HDRDIR = ./headers
SRCDIR = ./src

_OBJ = main.o  serialKnn.o pointsHelper.o
OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))

_DEPS = serialKnn.h pointsHelper.h
DEPS = $(patsubst %, $(HDRDIR)/%, $(_DEPS))


mainProgram: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $<  $(CFLAGS)

clean:
	rm -rf ./*.csv; rm -rf ./bin/*.o

clear: 
	rm -rf ./*.csv; rm -rf ./PythonCheck/*.csv