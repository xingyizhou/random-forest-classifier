all:
	g++ -fopenmp -O2 -w -o classificationforest common.h classificationforest.h classificationforest.cpp main.cpp 
