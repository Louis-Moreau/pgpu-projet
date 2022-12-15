CXX=g++
#CXXFLAGS=-O3 -march=native -I /usr/local/include/opencv4/ -lopencv_core -lopencv_videoio -lopencv_highgui
#GXXFLAGS=-O3 -I /usr/local/include/opencv4/ -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
CXXFLAGS=-O3 -march=native -I /opt/opencv/include/opencv4/ -lopencv_core -lopencv_videoio -lopencv_highgui
GXXFLAGS=-O3 -I /opt/opencv/include/opencv4/ -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
LDLIBS=`pkg-config --libs opencv4`
BIN=./bin/

all: sobel sobel1-cu sobel2-cu sobel3-cu sobel4-cu sobel4bis-cu

sobel: sobel.cpp
	$(CXX) $(CXXFLAGS) -o $(BIN)$@ $< $(LDLIBS)

sobel1-cu: sobel_1.cu
	nvcc $(GXXFLAGS) -o $(BIN)$@ $< $(LDLIBS)

sobel2-cu: sobel_2.cu
	nvcc $(GXXFLAGS) -o $(BIN)$@ $< $(LDLIBS)

sobel3-cu: sobel_3.cu
	nvcc $(GXXFLAGS) -o $(BIN)$@ $< $(LDLIBS)

sobel4-cu: sobel_4.cu
	nvcc $(GXXFLAGS) -o $(BIN)$@ $< $(LDLIBS)
	
sobel4bis-cu: sobel_4bis.cu
	nvcc $(GXXFLAGS) -o $(BIN)$@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm ./bin/*
	rm ./images/output/*