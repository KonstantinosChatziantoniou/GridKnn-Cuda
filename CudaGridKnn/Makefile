CC = nvcc
CFLAGS = -O2 --default-stream per-thread  -Xcompiler -fopenmp -lcuda -lcudart 
#-std=c99 --default-stream per-thread
LIBS = -lm #-Wall -Wextra
OBJDIR = ./bin
HDRDIR = ./headers
SRCDIR = ./src

_OBJ =  main.o serialKnn.o pointsHelper.o
OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))

_DEPS = serialKnn.h pointsHelper.h
DEPS = $(patsubst %, $(HDRDIR)/%, $(_DEPS))


mainProgram: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)


$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CC) -c -o $@ $<  $(CFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $<  $(CFLAGS)


#bin/cudaWrapper.o: src/cudaWrapper.cu headers/cudaWrapper.h
#	nvcc  -c -o bin/cudaWrapper.o src/cudaWrapper.cu 
clean:
	rm -rf ./*.csv; rm -rf ./bin/*.o

clear: 
	rm -rf ./*.csv; rm -rf ./PythonCheck/*.csv