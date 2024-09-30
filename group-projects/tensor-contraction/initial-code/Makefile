CXX        = nvcc -O3

SRC = tensor-main.cu
HELPERS = tensor-kernels.cu.h
EXECUTABLE = tensor-prod

default: compile run

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<

compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC)

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

