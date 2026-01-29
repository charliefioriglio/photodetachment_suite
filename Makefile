CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp

SRC_DIR = src/cxx

LIB_SOURCES = src/cxx/tools.cpp src/cxx/gauss.cpp src/cxx/shell.cpp src/cxx/molecule.cpp src/cxx/dyson.cpp src/cxx/grid.cpp src/cxx/math_special.cpp src/cxx/cross_section.cpp src/cxx/angle_grid.cpp src/cxx/rotation.cpp src/cxx/continuum.cpp src/cxx/beta.cpp src/cxx/num_eikr.cpp src/cxx/point_dipole.cpp src/cxx/clebsch_gordan.cpp src/cxx/physical_dipole.cpp
LIB_OBJECTS = $(LIB_SOURCES:.cpp=.o)

EXECUTABLES = dyson_gen beta_gen dipole_verifier continuum_plotter test_angular

all: $(EXECUTABLES)

dyson_gen: src/cxx/dyson_gen.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

beta_gen: src/cxx/beta_gen.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

dipole_verifier: src/cxx/dipole_verifier.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

continuum_plotter: src/cxx/continuum_plotter.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

test_angular: src/cxx/test_angular.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

test_radial: src/cxx/test_radial.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

compute_xs: src/cxx/compute_xs.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

verify_radial: src/cxx/verify_radial.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

verify_angular: src/cxx/verify_angular.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(SRC_DIR)/*.o $(EXECUTABLES) compute_xs verify_radial verify_angular
