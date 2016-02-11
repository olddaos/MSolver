# TODO: Change it to CMake
PROJECT := _caffe_facade

CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

##############################
# Derive include and lib directories
##############################

INCLUDE_DIRS += $(BUILD_INCLUDE_DIR) $(CAFFE_DIR)/include $(CAFFE_DIR)/src $(CAFFE_DIR)/build/src ./

CUDA_INCLUDE_DIR := $(CUDA_DIR)/include

CUDA_LIB_DIR :=
# add <cuda>/lib64 only if it exists
ifneq ("$(wildcard $(CUDA_DIR)/lib64)","")
	CUDA_LIB_DIR += $(CUDA_DIR)/lib64
endif
CUDA_LIB_DIR += $(CUDA_DIR)/lib
ifneq ($(CPU_ONLY), 1)
	INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
	LIBRARY_DIRS += $(CUDA_LIB_DIR)
	LIBRARIES := cudart cublas curand
endif
# cuDNN acceleration configuration.
ifeq ($(USE_CUDNN), 1)
	LIBRARIES += cudnn
	COMMON_FLAGS += -DUSE_CUDNN
endif
PYTHON_LIBRARIES := boost_python python2.7
WARNINGS := -Wall -Wno-sign-compare
STATIC_NAME := $(CAFFE_DIR)/build/lib/libcaffe.a

ifeq ($(USE_CUDNN), 1)
	LIBRARIES += cudnn
	COMMON_FLAGS += -DUSE_CUDNN
endif
##############################
# Configure build
##############################

CXX ?= /usr/bin/g++
GCCVERSION := $(shell $(CXX) -dumpversion | cut -f1,2 -d.)
# older versions of gcc are too dumb to build boost with -Wuninitalized
ifeq ($(shell echo $(GCCVERSION) \< 4.6 | bc), 1)
  WARNINGS += -Wno-uninitialized
endif

LIBRARIES += glog gflags protobuf leveldb snappy lmdb \
	     boost_system hdf5_hl hdf5 opencv_core opencv_highgui opencv_imgproc\
	     pthread boost_thread boost_python python2.7 #opencv_imgcodecs

BLAS ?= atlas
LINUX ?= 1
ifeq ($(BLAS), mkl)
	# MKL
	LIBRARIES += mkl_rt
	COMMON_FLAGS += -DUSE_MKL
	MKL_DIR ?= /opt/intel/mkl
	BLAS_INCLUDE ?= $(MKL_DIR)/include
	BLAS_LIB ?= $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64
else ifeq ($(BLAS), open)
	# OpenBLAS
	LIBRARIES += openblas
else
	# ATLAS
	ifeq ($(LINUX), 1)
		ifeq ($(BLAS), atlas)
			# Linux simply has cblas and atlas
			LIBRARIES += cblas atlas
		endif
	else ifeq ($(OSX), 1)
		# OS X packages atlas as the vecLib framework
		BLAS_INCLUDE ?= /System/Library/Frameworks/vecLib.framework/Versions/Current/Headers/
		LIBRARIES += cblas
		LDFLAGS += -framework vecLib
	endif
endif

# CPU-only configuration
ifeq ($(CPU_ONLY), 1)
	COMMON_FLAGS += -DCPU_ONLY
endif
LIBRARY_DIRS += $(BLAS_LIB)

ifeq ($(WITH_REDIS),1)
	LIBRARIES += redis3m
endif
# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)
# mex may invoke an older gcc that is too liberal with -Wuninitalized
LINKFLAGS += -fPIC $(COMMON_FLAGS) $(WARNINGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library))
PYTHON_LDFLAGS := $(LDFLAGS) $(foreach library,$(PYTHON_LIBRARIES),-l$(library))

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
else
	COMMON_FLAGS += -DNDEBUG -O2
endif
##############################
# Define build targets
##############################
.PHONY: py

all: py

py:
	$(CXX) -shared -o $(PROJECT).so _caffe.cpp -Wl,--whole-archive $(STATIC_NAME) \
	-Wl,--no-whole-archive -fPIC $(COMMON_FLAGS) \
	-Wall -Wno-sign-compare $(LDFLAGS)
	@ echo
tests:
	$(CXX) -o test_$(PROJECT) ../../../engine/caffe/src/gtest/gtest-all.cpp tests.cpp _caffe.cpp -Wl,--whole-archive $(STATIC_NAME) \
	-Wl,--no-whole-archive -fPIC $(COMMON_FLAGS) -I$(GTEST_DIR)	\
	-Wall -Wno-sign-compare $(LDFLAGS)


clean:
	@- $(RM) $(PROJECT).so test_$(PROJECT)
