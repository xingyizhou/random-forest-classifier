CXX ?= g++
CC ?= gcc
CFLAGS = -O3 -w
SHVER = 3
OS = $(shell uname)

all: train predict

train: classificationforest.o train.cpp 
	$(CXX) $(CFLAGS) -o train train.cpp classificationforest.o

predict: classificationforest.o predict.cpp 
	$(CXX) $(CFLAGS) -o predict predict.cpp classificationforest.o

common.o: common.h
	$(CXX) $(CFLAGS) -c -o common.o common.h

classificationforest.o: classificationforest.cpp classificationforest.h
	$(CXX) $(CFLAGS) -c -o classificationforest.o classificationforest.cpp

clean:
	rm -f *~ common.o classificationforest.o predict train *.model *.output
