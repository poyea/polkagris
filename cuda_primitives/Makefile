# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

NVCC ?= nvcc
ARCH ?= -arch=native
NVCCFLAGS ?= -O2 -x cu

ifeq ($(OS),Windows_NT)
EXT := .exe
RUN :=
RM := del /q
else
EXT :=
RUN := ./
RM := rm -f
endif

all: vector_add_test$(EXT) naive_matmul_test$(EXT)

vector_add_test$(EXT): host_vector_add.cpp vector_add.cu
	$(NVCC) $(ARCH) $(NVCCFLAGS) -o $@ $^

naive_matmul_test$(EXT): host_naive_matmul.cpp naive_matmul.cu
	$(NVCC) $(ARCH) $(NVCCFLAGS) -o $@ $^

run: all
	$(RUN)vector_add_test$(EXT)
	$(RUN)naive_matmul_test$(EXT)

clean:
	$(RM) vector_add_test$(EXT) naive_matmul_test$(EXT)
