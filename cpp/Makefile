HOSTNAME := $(shell hostname)
CXX = g++
CXXFLAGS = -std=c++14 -O3 -DPINOCCHIO_WITH_URDFDOM -fPIC

# Get the prefix of the active Conda environment.
CONDA_PREFIX ?= $(shell conda info --json | python -c "import sys, json; print(json.load(sys.stdin)[\"active_prefix\"])")

INCLUDE_DIRS = -I$(CONDA_PREFIX)/include/eigen3 \
               -I$(CONDA_PREFIX)/include/python3.11 \
			   -I$(CONDA_PREFIX)/include \
			   -Iinclude
PYTHON_INCLUDE = $(shell python3-config --includes)
PYTHON_SUFFIX = $(shell python3-config --extension-suffix)
LIB_DIRS = -L$(CONDA_PREFIX)/lib

LIBS = -lpinocchio -lboost_filesystem -lboost_serialization -lboost_system \
       -lurdfdom_sensor -lurdfdom_model_state -lurdfdom_model -lurdfdom_world -lconsole_bridge
  
aghf_pybind: aghf_pybind.cpp src/PybindWrapper.cpp src/ActivatedSphereAvoidancePenalty.cpp src/Spatial.cpp src/ActivatedStateLimitsPenalty.cpp src/LowerDifferenceInequalityConstraint.cpp src/TanhActivation.cpp src/UpperDifferenceInequalityConstraint.cpp src/SphereInequalityConstraint.cpp
	$(CXX) $(CXXFLAGS) $(PYTHON_INCLUDE) $(INCLUDE_DIRS) $(LIB_DIRS) -shared -o $@$(PYTHON_SUFFIX) $^ $(LIBS)
	mv aghf_pybind$(PYTHON_SUFFIX) ../src/phlame/aghf_pybind.so

clean:
	rm -f $(TARGET)
