#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../headers/pointsHelper.h"

float generatePoints(float* points, int numberOfPoints, int dimensions, float min, float max, int seed){
    srand(seed);
    for(int i = 0; i < numberOfPoints; i++){
        for(int j = 0; j < dimensions; j++){
            points[i*dimensions + j] = min + (max-min)*((float)rand()/(float)RAND_MAX);
        }
    }
}

void printPoints(float *points, int numberOfPoints, int dimensions){
    for(int i = 0; i < numberOfPoints; i++){
        printf("Point%d\t",i);
        for(int j = 0; j < dimensions; j++){
            printf("x%d %f\t",j,points[i*dimensions+j]);
        }
        printf("\n");
    }
}

void printPointsToCsv(const char* name,const char* mode ,float* points, int numberOfPoints, int dimensions){
    FILE* file = fopen(name,mode);
    for(int i = 0; i < numberOfPoints; i++){
        for(int j = 0; j < dimensions-1; j++){
            fprintf(file,"%f,",points[i*dimensions+j]);
        }
        fprintf(file,"%f\n",points[i*dimensions + dimensions -1]);
    }
    fclose(file);
}

float distanceEucl(float *p1, float* p2, int dimensions){
    float dist = 0;
    for(int i = 0; i < dimensions; i++){
        dist += pow(p1[i] - p2[i] , 2);
    }
    dist = sqrt(dist);
    return dist;
}

float distanceManh(float *p1, float* p2, int dimensions){
    float dist = 0;
    for(int i = 0; i < dimensions; i++){
        dist += abs(p1[i] - p2[i]);
    }
    return dist;
}