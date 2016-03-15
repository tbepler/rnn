CC := gcc
FC := gfortran

USE_MKL := true

BLAS := -lopenblas
MATH := -lm
OMP := -lgomp

MKLROOT := /opt/intel/mkl
MKL :=  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_gf_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -ldl -lpthread -lm

FFLAGS := -Wall -Wextra -cpp -ffree-form -std=f2008 -O3 -fpic -fopenmp -march=native -mtune=native -ftree-vectorize -ftree-vectorizer-verbose=8 -fopt-info-vec-optimized -mfpmath=sse -ffast-math -fall-intrinsics
CFLAGS := -Wall -Wextra -pedantic -std=c11 -O3 -fpic -fopenmp -march=native -ffast-math -ftree-vectorizer-verbose=8 -fopt-info-vec-optimized
LDFLAGS := -fopenmp

ifeq ($(USE_MKL),true)
	LINK := $(MKL) $(OMP) -lgfortran
else
	LINK := $(BLAS) $(MATH) $(OMP) -lgfortran	
endif

SRC_DIR := src
OBJ_DIR := obj
LIB_DIR := lib

CSRCS := $(shell find -L $(SRC_DIR) -name "*.c")
FSRCS := $(shell find -L $(SRC_DIR) -name "*.f")
OBJS := $(CSRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o) $(FSRCS:$(SRC_DIR)/%.f=$(OBJ_DIR)/%.o)

.PHONY: clean all lib profile

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.f $(SRC_DIR)/%.fi
	@mkdir -p $(OBJ_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(LIB_DIR)/librnn.so: $(OBJS)
	@mkdir -p $(LIB_DIR)
	$(CC) $(LDFLAGS) -shared $^ $(LINK) -o $@

profile-openblas:
	gcc -std=c11 -O3 -ftree-vectorizer-verbose=8 -fopt-info-vec-optimized -march=native -c src/lstm-batched.c
	gcc -std=c11 -c lstm_profile.c
	$(FC) $(FFLAGS) -ftree-vectorizer-verbose=8 -fopt-info-vec-optimized -c src/lstm.f
	gcc lstm-batched.o lstm.o lstm_profile.o $(BLAS) $(MATH) -lgfortran -o lstm-profile.out

profile-mkl:
	gcc $(CFLAGS) -ftree-vectorizer-verbose=8 -fopt-info-vec-optimized -c src/lstm-batched.c
	gcc -std=c11 -c lstm_profile.c
	$(FC) $(FFLAGS) -ftree-vectorizer-verbose=8 -fopt-info-vec-optimized -c src/lstm.f
	gcc lstm-batched.o lstm.o lstm_profile.o $(MKL) $(OMP) -lgfortran -o lstm-profile.out

clean:
	@rm -f $(OBJS)
	@if [ -d $(OBJ_DIR) ]; then rmdir $(OBJ_DIR); fi
	@rm -f $(LIB_DIR)/librnn.so
	@if [ -d $(LIB_DIR) ]; then rmdir $(LIB_DIR); fi
