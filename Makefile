CUDA_PATH = /usr/local/cuda-12.6
CXX = $(CUDA_PATH)/bin/nvcc
CXXFLAGS = -w
ifdef PRINT_RNORM
	CXXFLAGS += -DPRINT_RNORM
endif
EIGEN_DIR = /usr/include/eigen-3.4.0
LIBS = -lmatio -lcublas -lcusparse -lcusolver

SRC = mp_agmres.cu gmres.cu mp_gmres.cu agmres.cu
OBJ = $(SRC:.cu=.o)

OUT = mp_agmres.o gmres.o mp_gmres.o agmres.o

all: $(OUT)

%.o: %.cu
	$(CXX) $(CXXFLAGS) $< -o $@ -I $(EIGEN_DIR) $(LIBS)

clean:
	rm -f $(OBJ) $(OUT)

.PHONY: all clean
