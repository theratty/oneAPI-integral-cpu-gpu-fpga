CXX = dpcpp
CXXFLAGS = -O0 -std=c++17

EXE_NAME = out
SRC = main.cpp
OBJ = main.o

$(EXE_NAME): $(SRC)
	$(CXX) $(CXXFLAGS) -fintelfpga $(SRC) -o $(EXE_NAME) -Xshardware

link: $(EXE_NAME)

run: $(EXE_NAME)
	./$(EXE_NAME)

clean:
	rm -f $(EXE_NAME) $(OBJ) build.sh.* run.sh.*

.PHONY: clean, build, link
