#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../headers/pointsHelper.h"
#include "../headers/serialKnn.h"

/* Internal Declarations */




void assignPointsToBlocks(float *points, int* block_of_point, int* points_per_block, float side_length, int number_of_points, int grid_d, int dimensions){
    for(int i = 0; i < grid_d*grid_d*grid_d; i++){
        points_per_block[i] = 0;
    }

    for(int i = 0; i < number_of_points; i++){
        for(int j = 0; j < dimensions; j++){
            block_of_point[i*dimensions + j] = (int)floor(points[i*dimensions+j]/side_length);
        }
        points_per_block[grid_d*grid_d*block_of_point[i*dimensions] + grid_d*block_of_point[i*dimensions+1] + block_of_point[i*dimensions+2]]++;
    }

    for(int i = 0; i < grid_d; i++){
        for(int j = 0 ; j < grid_d; j++){
            for(int k = 0; k < grid_d; k++){
                //printf("block%d%d%d %d points => %d\n",i,j,k,i*grid_d*grid_d + j*grid_d + k,points_per_block[i*grid_d*grid_d + j*grid_d + k]);
            }
        }
    }

}


void rearrangePointsToGrid(float *points,float* gird_of_points,
                            int* block_of_point, int* points_per_block, 
                            float side_length, int number_of_points, 
                            int grid_d, int dimensions)
{

    int* stack_counter = malloc(grid_d*grid_d*grid_d*sizeof(int));
    int* integral_points_per_block = malloc(grid_d*grid_d*grid_d*sizeof(int));

    for(int i = 0; i < grid_d*grid_d*grid_d; i++){
        stack_counter[i] = 0;
        integral_points_per_block[i] = 0;
        for(int j = 0; j < i; j++){
            integral_points_per_block[i] += points_per_block[j];
        }
    }

    for(int i = 0; i < number_of_points; i++){
        int gridx,gridy,gridz;
        gridx = block_of_point[i*dimensions];
        gridy = block_of_point[i*dimensions + 1];
        gridz = block_of_point[i*dimensions + 2];
        int gridIndex = gridx*grid_d*grid_d + gridy*grid_d + gridz;
        int index = integral_points_per_block[gridIndex];
        memcpy(&gird_of_points[index*dimensions+dimensions*stack_counter[gridIndex]], &points[i*dimensions], dimensions*sizeof(float));
        stack_counter[gridIndex]++;
    }

    free(stack_counter);
    free(integral_points_per_block);
}

void searchKnnInBlock(float* points_to_search, int number_of_points_to_search,
                    float* query, int dimensions,
                    float* knn, float* knns_dist)
{
    for(int p = 0; p < number_of_points_to_search; p++){
        float temp_dist = distanceEucl(query, &points_to_search[p*dimensions], dimensions);
        if(temp_dist < *knns_dist){
            memcpy(knn,&points_to_search[p*dimensions],dimensions*sizeof(float));
            *knns_dist = temp_dist;
        }
    }


}

void addNeighbour(kNeighbours* kn, float* point, float dist, int dimensions){
    //if the array of neighbours is not full, keep adding//
    if(kn->num_of_nbrs < kn->k){
        memcpy(&(kn->nb_points[kn->num_of_nbrs*dimensions]), point ,dimensions*sizeof(float));
        kn->dists[kn->num_of_nbrs] = dist;
        kn->num_of_nbrs++;
        //printf("added points with non full nblist n = %d,  ",kn->num_of_nbrs-1);
        //printPoints(point,1,dimensions);
        if(kn->num_of_nbrs == kn->k){
            kn->max_dist = -1;
            for(int i = 0; i < kn->k; i++){
                if(kn->dists[i] > kn->max_dist){
                    kn->max_dist = kn->dists[i];
                    kn->max_index = i;
                }
            }
        }
        return;
    }

    if(dist < kn->max_dist){
        //printf("max dist was %f, new is %f\n",kn->max_dist,dist);
        memcpy(&(kn->nb_points[kn->max_index*dimensions]), point ,dimensions*sizeof(float));
        kn->dists[kn->max_index] = dist;
        kn->max_dist = -1;
        for(int i = 0; i < kn->k; i++){
            if(kn->dists[i] > kn->max_dist){
                kn->max_dist = kn->dists[i];
                kn->max_index = i;
            }
        }
    }
    return;

}









void getNeighbourBlocks(NeighbourBlocks* nb,int grid_d, int* gridID){
    int maxNbrs = 3*3*3-1;
    int* temp = malloc(3*maxNbrs*sizeof(int));
    int index = 0;
    for(int i = -1; i <= 1; i++){
        for(int j = -1; j <= 1; j++){
            for(int k = -1; k <= 1; k++){
                if(i != 0 | j != 0 | k != 0){
                    int nx = gridID[0] + i;
                    int ny = gridID[1] + j;
                    int nz = gridID[2] + k;

                    if(!(nx<0 | ny<0 | nz<0 | nx >=grid_d | ny >= grid_d | nz>=grid_d)){
                        temp[index*3] = nx;
                        temp[index*3 + 1] = ny;
                        temp[index*3 + 2] = nz;
                        index++;
                    }
                }
            }
        }
    }

    nb->num_of_nbr_blocks = index;
    nb->nbr_blocks = malloc(index*3*sizeof(int));
    memcpy(nb->nbr_blocks , temp , index*3*sizeof(int));
    free(temp);
}


