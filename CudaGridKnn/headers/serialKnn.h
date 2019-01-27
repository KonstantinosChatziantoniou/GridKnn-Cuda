#ifndef SERIALKNN_H
#define SERIALKNN_H


#ifdef __cplusplus
extern "C" 
#endif 
typedef struct{
    int* nbr_blocks;
    int num_of_nbr_blocks;
}NeighbourBlocks;

#ifdef __cplusplus
extern "C" 
#endif 
typedef struct{
    float max_dist;
    int max_index;
    float* dists;
    float* nb_points;
    int num_of_nbrs;
    int k;
}kNeighbours;


#ifdef __cplusplus
extern "C" 
#endif 
void assignPointsToBlocks(float *points,
                        int* block_of_point,
                        int* points_per_block,
                        float side_length,
                        int number_of_points,
                        int grid_d,
                        int dimensions);

#ifdef __cplusplus
extern "C" 
#endif 
void rearrangePointsToGrid(float *points,float* gird_of_points,
                            int* block_of_point, int* points_per_block, 
                            float side_length, int number_of_points, 
                            int grid_d, int dimensions);


#ifdef __cplusplus
extern "C" 
#endif 
void searchKnnInBlock(float* points_to_search, int number_of_points_to_search,
                    float* query, int dimensions,
                    float* knn, float* knns_dist);



#ifdef __cplusplus
extern "C" 
#endif 
void getNeighbourBlocks(NeighbourBlocks* nb,int grid_d, int* gridID);


#ifdef __cplusplus
extern "C" 
#endif 
void addNeighbour(kNeighbours* kn, float* point, float dist, int dimensions);

#ifdef __cplusplus
extern "C" 
#endif 
void parrallelAssignPointsToBlocks(float *points,
                                     int* block_of_point, 
                                     int* points_per_block, 
                                     float side_length, 
                                     int number_of_points, 
                                     int grid_d, 
                                     int dimensions);










#endif