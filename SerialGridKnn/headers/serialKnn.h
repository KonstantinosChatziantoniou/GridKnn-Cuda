#ifndef SERIALKNN_H
#define SERIALKNN_H


typedef struct{
    int* nbr_blocks;
    int num_of_nbr_blocks;
}NeighbourBlocks;

typedef struct{
    float max_dist;
    int max_index;
    float* dists;
    float* nb_points;
    int num_of_nbrs;
    int k;
}kNeighbours;


void assignPointsToBlocks(float *points,
                        int* block_of_point,
                        int* points_per_block,
                        float side_length,
                        int number_of_points,
                        int grid_d,
                        int dimensions);

void rearrangePointsToGrid(float *points,float* gird_of_points,
                            int* block_of_point, int* points_per_block, 
                            float side_length, int number_of_points, 
                            int grid_d, int dimensions);


void searchKnnInBlock(float* points_to_search,
                    int number_of_points_to_search,
                    float* query,
                    int dimensions,
                    kNeighbours* knn_str);



void getNeighbourBlocks(NeighbourBlocks* nb,int grid_d, int* gridID);


void addNeighbour(kNeighbours* kn, float* point, float dist, int dimensions);

#endif