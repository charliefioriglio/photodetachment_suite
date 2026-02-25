CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp

code_DIR = code/cxx

LIB_SOURCES = code/cxx/tools.cpp code/cxx/gauss.cpp code/cxx/shell.cpp code/cxx/molecule.cpp code/cxx/dyson.cpp code/cxx/grid.cpp code/cxx/math_special.cpp code/cxx/cross_section.cpp code/cxx/angle_grid.cpp code/cxx/rotation.cpp code/cxx/continuum.cpp code/cxx/beta.cpp code/cxx/num_eikr.cpp code/cxx/point_dipole.cpp code/cxx/clebsch_gordan.cpp code/cxx/physical_dipole.cpp
LIB_OBJECTS = $(LIB_SOURCES:.cpp=.o)

EXECUTABLES = dyson_gen beta_gen dipole_verifier continuum_plotter test_angular

all: $(EXECUTABLES)

dyson_gen: code/cxx/dyson_gen.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

beta_gen: code/cxx/beta_gen.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

dipole_verifier: code/cxx/dipole_verifier.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

continuum_plotter: code/cxx/continuum_plotter.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

test_angular: code/cxx/test_angular.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

test_radial: code/cxx/test_radial.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

compute_xs: code/cxx/compute_xs.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

verify_radial: code/cxx/verify_radial.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

verify_angular: code/cxx/verify_angular.o $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(code_DIR)/*.o $(EXECUTABLES) compute_xs verify_radial verify_angular
