all: main.cc
	g++ main.cc -lglog -o cuda_cholesky_test -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver

clean:
	rm -f cuda_cholesky_test