CXX ?= g++
CC ?= gcc
CFLAGS = -O3 -w
SHVER = 3
OS = $(shell uname)

all: train predict

train: common.o classificationforest.o train.cpp 
	$(CXX) $(CFLAGS) -o train train.cpp common.o classificationforest.o

predict: common.o classificationforest.o predict.cpp 
	$(CXX) $(CFLAGS) -o predict predict.cpp common.o classificationforest.o

common.o: common.h
	$(CXX) $(CFLAGS) -c -o common.o common.h

classificationforest.o: classificationforest.cpp classificationforest.h
	$(CXX) $(CFLAGS) -c -o classificationforest.o classificationforest.cpp

clean:
	rm -f *~ common.o classificationforest.o predict train *.model *.output
