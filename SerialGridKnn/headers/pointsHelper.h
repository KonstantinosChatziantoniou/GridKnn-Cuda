#ifndef PT_HELPER_H
#define PT_HELPER_H

float generatePoints(float* points, int numberOfPoints, int dimensions, float min, float max, int seed);

void printPoints(float *points, int numberOfPoints, int dimensions);

void printPointsToCsv(const char* name,const char* mode ,float* points, int numberOfPoints, int dimensions);

float distanceEucl(float *p1, float* p2, int dimensions);

float distanceManh(float *p1, float* p2, int dimensions);

#endif