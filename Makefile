CXX=nvcc
CUX=$(wildcard src/*.cu)
OBJ=$(CUX:.cu=.o)
DEPS=$(wildcard include/*.h)
FLAGS=-arch=sm_70 -g

%.o: %.cu $(DEPS)
	$(CXX) $(FLAGS) -dc -o $@ $< -Iinclude

ntt_cuda: $(OBJ) 
	$(CXX) $(FLAGS) --output-file $@ $^

clean:
	rm -rfv ntt ntt_cuda $(OBJ)
