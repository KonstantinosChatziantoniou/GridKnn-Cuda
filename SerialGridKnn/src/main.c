#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#include "../headers/pointsHelper.h"
#include "../headers/serialKnn.h"

/******************** INPUT *********************
 * 1st param -> number of points    (default 2^5)
 * 2nd param -> grid dimensions     (default 2^1)
 * 3rd param -> seed                (default 1,2)
*************************************************/
int main(int argc, char** argv){
    /*****For Time Measuring*****/
    struct timeval tstart,tend;
    gettimeofday(&tstart,NULL);

    int number_of_points = 5;
    int grid_d = 1;
    int k_num = 2;
    int seed = 1;
    if(argc > 1){
        number_of_points = atoi(argv[1]);
    }
    number_of_points = pow(2,number_of_points);
    if(argc > 2){
        grid_d = atoi(argv[2]);
    }
    grid_d = pow(2,grid_d);
    if(argc > 3){
        k_num = atoi(argv[3]);
    }
    if(argc > 4){
        seed = atoi(argv[4]);
    }
    int number_of_queries = number_of_points;
    int dimensions = 3;
    float side_block_length = ((float)1)/((float)grid_d);
    printf("Number of points:%d\nNumber of queries:%d\nDimensions:%d\nGrid Dimensions:%d\nK for k-nn:%d\nSideBlock Length%f\n",
                                                number_of_points,number_of_queries,dimensions,grid_d,k_num,side_block_length);

    float* points = malloc(number_of_points*dimensions*sizeof(float));
    float* queries = malloc(number_of_queries*dimensions*sizeof(float));
    float* grid_arranged_points = malloc(number_of_points*dimensions*sizeof(float));
    float* grid_arranged_queries = malloc(number_of_queries*dimensions*sizeof(float));
    int* block_of_point = malloc(number_of_points*dimensions*sizeof(int));
    int* block_of_query = malloc(number_of_queries*dimensions*sizeof(int));
    int* points_per_block = malloc(grid_d*grid_d*grid_d*sizeof(int));
    int* queries_per_block = malloc(grid_d*grid_d*grid_d*sizeof(int));
    int* integral_points_per_block = malloc(grid_d*grid_d*grid_d*sizeof(int));
    int* integral_queries_per_block = malloc(grid_d*grid_d*grid_d*sizeof(int));
    
    kNeighbours* knn_struct = malloc(number_of_queries*sizeof(kNeighbours));
    for(int q = 0; q < number_of_queries; q++){
        knn_struct[q].nb_points = malloc(k_num*dimensions*sizeof(float));
        knn_struct[q].dists = malloc(k_num*sizeof(float));
        knn_struct[q].max_dist = 100;
        knn_struct[q].max_index = -1;
        knn_struct[q].num_of_nbrs = 0;
        knn_struct[q].k = k_num;
    }

    //printf("mallocs done\n");
    generatePoints(points, number_of_points, dimensions, 0, 1, 1);
    generatePoints(queries, number_of_queries, dimensions, 0, 1, 2);

    



    assignPointsToBlocks(points, block_of_point , points_per_block , side_block_length , number_of_points, grid_d , dimensions);
    //printf("queries--\n");
    assignPointsToBlocks(queries, block_of_query , queries_per_block , side_block_length , number_of_queries, grid_d , dimensions);

    for(int i = 0; i < grid_d*grid_d*grid_d; i++){
        integral_points_per_block[i] = 0;
        integral_queries_per_block[i] = 0;
        for(int j = 0; j < i; j++){
            integral_points_per_block[i] += points_per_block[j];
            integral_queries_per_block[i] += queries_per_block[j];
        }
    }

    for(int i = 0; i < grid_d*grid_d*grid_d; i++){
        //printf("block %d contains %d\n",i,integral_points_per_block[i]);
    }   

    for(int i = 0; i < grid_d*grid_d*grid_d; i++){

        //printf("block %d contains %d\n",i,integral_queries_per_block[i]);
    }


    /*
    for(int q = 0; q < number_of_queries; q++){
        for(int p = 0; p < number_of_points; p++){
            float tempdist = distanceEucl(&points[p*dimensions] , &queries[q*dimensions] , dimensions);
            addNeighbour(&knn_struct[q], &points[p*dimensions], tempdist, dimensions);
        }
    }*/







    
    rearrangePointsToGrid(points,grid_arranged_points, block_of_point , points_per_block , side_block_length , number_of_points, grid_d , dimensions);
    rearrangePointsToGrid(queries,grid_arranged_queries, block_of_query , queries_per_block , side_block_length , number_of_queries, grid_d , dimensions);

    assignPointsToBlocks(grid_arranged_points, block_of_point , points_per_block , side_block_length , number_of_points, grid_d , dimensions);
    assignPointsToBlocks(grid_arranged_queries, block_of_query , queries_per_block , side_block_length , number_of_queries, grid_d , dimensions);

    for(int i = 0; i < number_of_points; i++){
        //printf("point %d block->",i);
        for(int j = 0; j < dimensions; j++){
            //printf("[%d] ",block_of_point[i*dimensions+j]);
        }//printf("\n");
    }
    for(int i = 0; i < number_of_queries; i++){
        //printf("query %d block->",i);
        for(int j = 0; j < dimensions; j++){
            //printf("[%d] ",block_of_query[i*dimensions+j]);
        }//printf("\n");
    }

    // KNN search (basic candidates) 
    for(int q = 0; q < number_of_queries; q++){
        int* currentBlock = malloc(dimensions*sizeof(int));
        //printf("current block%d ",q);
        for(int i = 0; i < dimensions; i++){
            currentBlock[i] = block_of_query[q*dimensions + i];
            //printf("[%d] ",currentBlock[i]);
        }//printf("\n");
        int gridIndex = 0;
        for(int i = 0; i < dimensions; i++){
            gridIndex += pow(grid_d,dimensions -1 -i)*currentBlock[i];
        }
        int startingAddressOfPoints = integral_points_per_block[gridIndex];
        int numberOfPointsInBlock = points_per_block[gridIndex];
        //printf("starting index %d, num of points %d, grindex %d\n",startingAddressOfPoints,numberOfPointsInBlock,gridIndex);
        float* pointToSearch = &grid_arranged_points[startingAddressOfPoints*dimensions];
        searchKnnInBlock(pointToSearch,numberOfPointsInBlock,&grid_arranged_queries[q*dimensions], dimensions , &knn_struct[q]);
        //printf("Nbs of query: ");
        //printPoints(&grid_arranged_queries[q*dimensions],1,dimensions);
        //printf("----Neighbours----\n");
        //printPoints(knn_struct[q].nb_points, k_num , dimensions);
        //printf("---------------------------\n");
        // printf("Nbs of query: ");
        //         printPoints(&grid_arranged_queries[q*dimensions],1,dimensions);
        //         printf("----Neighbours----\n");
        //         printPoints(knn_struct[q].nb_points, k_num , dimensions);
        //         printf("----Dists----\n");
        //         printPoints(knn_struct[q].dists,k_num,1);
        //         printf("---------------------------\n");
    }
    int counterOutCand = 0;
    // Knn NeighbourBlocks 
    for(int q = 0; q < number_of_queries; q++){
        float max_dist = knn_struct[q].max_dist;
        float min_dist_from_bounds = 1000;
        for(int i = 0; i < dimensions; i++){
            float tempdist = fmod(grid_arranged_queries[q*dimensions + i],side_block_length);
            float tempdist2 = side_block_length - tempdist;
            if(tempdist2 < tempdist){
                tempdist = tempdist2;
            }

            if(tempdist < min_dist_from_bounds){
                min_dist_from_bounds = tempdist;
            }
        }
        if(max_dist > min_dist_from_bounds){
            counterOutCand++;
            NeighbourBlocks nb;
            getNeighbourBlocks(&nb,grid_d,&block_of_query[q*dimensions]);
            for(int b = 0; b < nb.num_of_nbr_blocks; b++){
                //printf("blocks of %d/%d [%d,%d,%d]\n",q,b,nb.nbr_blocks[b*dimensions],nb.nbr_blocks[b*dimensions+1],nb.nbr_blocks[b*dimensions+2]);
            }
            for(int b = 0; b < nb.num_of_nbr_blocks; b++){
                int* currentBlock = malloc(dimensions*sizeof(int));
                for(int i = 0; i < dimensions; i++){
                    currentBlock[i] = nb.nbr_blocks[b*dimensions + i];
                }
                int gridIndex = 0;
                for(int i = 0; i < dimensions; i++){
                    gridIndex += pow(grid_d,dimensions -1 -i)*currentBlock[i];
                }
                //printf("current block222  %d/%d ",q,b);
                for(int i = 0; i < dimensions; i++){
                    //printf("[%d] ",currentBlock[i]);
                }//printf("\n");
                int startingAddressOfPoints = integral_points_per_block[gridIndex];
                int numberOfPointsInBlock = points_per_block[gridIndex];
                float* pointToSearch = &grid_arranged_points[startingAddressOfPoints*dimensions];
                searchKnnInBlock(pointToSearch,numberOfPointsInBlock,&grid_arranged_queries[q*dimensions], dimensions , &knn_struct[q]);
                // printf("Nbs of query: ");
                // printPoints(&grid_arranged_queries[q*dimensions],1,dimensions);
                // printf("----Neighbours----\n");
                // printPoints(knn_struct[q].nb_points, k_num , dimensions);
                // printf("----Dists----\n");
                // printPoints(knn_struct[q].dists,k_num,1);
                // printf("---------------------------\n");
            }
        }
    }
    
    
    float* knn_res = malloc(number_of_queries*k_num*dimensions*sizeof(float));
    for(int q = 0; q < number_of_queries; q++){
        //printPoints(knn_struct[q].nb_points, k_num , dimensions);
        memcpy(&knn_res[q*dimensions*k_num] , knn_struct[q].nb_points , k_num*dimensions*sizeof(float));
    }
    //Time measuring (end)//
    gettimeofday(&tend,NULL);
    printf("Time elapsed %ld s, %ld us\n",tend.tv_sec - tstart.tv_sec, tend.tv_usec - tstart.tv_usec);
    printf("%f %% searched other boxes (%d/%d)\n  ",100*(float)counterOutCand/number_of_queries,counterOutCand,number_of_queries);
    printPointsToCsv("knn.csv" , "w" , knn_res , number_of_queries*k_num , dimensions);
    printPointsToCsv("points.csv" , "w" , points , number_of_points , dimensions);
    printPointsToCsv("queries.csv" , "w" , queries , number_of_queries , dimensions);
    printPointsToCsv("points_arranged.csv" ,"w" , grid_arranged_points , number_of_points , dimensions);
    printPointsToCsv("queries_arranged.csv" , "w" , grid_arranged_queries , number_of_queries , dimensions);

    free(points);
    free(queries);
    free(grid_arranged_points);
    free(grid_arranged_queries);
    free(block_of_point);
    free(block_of_query);
    free(points_per_block);
    free(queries_per_block);

    for(int q = 0; q < number_of_queries; q++){
        free(knn_struct[q].nb_points);
        free(knn_struct[q].dists);
    }
    free(knn_struct);
    free(knn_res);
    
    
    return 0;
}