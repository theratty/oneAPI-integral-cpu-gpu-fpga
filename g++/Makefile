CXX = g++
CXXFLAGS = -O0 -std=c++17

EXE_NAME = out
SRC = main.cpp
OBJ = main.o

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -c -o $(OBJ)

$(EXE_NAME): $(OBJ)
	$(CXX) $(OBJ) -o $(EXE_NAME)

build: $(OBJ)
link: $(EXE_NAME)

run: $(EXE_NAME)
	./$(EXE_NAME)

clean:
	rm -f $(EXE_NAME) $(OBJ) job.sh.*

submit:
	qsub -l nodes=1:gpu:ppn=2 -d . job.sh

.PHONY: clean, build, link, submit
