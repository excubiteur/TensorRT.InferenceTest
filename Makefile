PKGS:= opencv4
CXX=gcc
CXXFLAGS+= -I/usr/local/cuda/include $(shell pkg-config --cflags $(PKGS))
LIBS := -lstdc++ \
  -L/usr/local/cuda/lib64 -lcudart -lnvinfer_plugin -lnvcaffe_parser -lnvinfer
LIBS+=$(shell pkg-config --libs $(PKGS))

all: Serialize.exe InferFromEngine.exe

Serialize.exe: Serialize.o 
	$(CXX) -o Serialize.exe Serialize.o $(LIBS)

InferFromEngine.exe: InferFromEngine.o 
	$(CXX) -o InferFromEngine.exe InferFromEngine.o $(LIBS)

clean:
	rm -f *.exe
	rm -f *.o
	rm -f *.bin

	
