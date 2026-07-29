# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

NVCC ?= nvcc
ARCH ?= -arch=native
NVCCFLAGS ?= -O2
SIZE ?= 2048

ifeq ($(OS),Windows_NT)
EXT := .exe
RUN :=
RM := del /q
else
EXT :=
RUN := ./
RM := rm -f
endif

all: bench_sgemm$(EXT)

bench_sgemm$(EXT): bench_sgemm.cu sgemm.cu sgemm.cuh
	$(NVCC) $(ARCH) $(NVCCFLAGS) -o $@ bench_sgemm.cu sgemm.cu -lcublas

run: all
	$(RUN)bench_sgemm$(EXT) $(SIZE)

clean:
	$(RM) bench_sgemm$(EXT)
