#ifndef PT_HELPER_H
#define PT_HELPER_H

#ifdef __cplusplus
extern "C" 
#endif 
float generatePoints(float* points, int numberOfPoints, int dimensions, float min, float max, int seed);

#ifdef __cplusplus
extern "C" 
#endif 
void printPoints(float *points, int numberOfPoints, int dimensions);

#ifdef __cplusplus
extern "C" 
#endif 
void printPointsToCsv(const char* name,const char* mode ,float* points, int numberOfPoints, int dimensions);

#ifdef __cplusplus
extern "C" 
#endif 
float distanceEucl(float *p1, float* p2, int dimensions);

#ifdef __cplusplus
extern "C" 
#endif 
float distanceManh(float *p1, float* p2, int dimensions);

#endif