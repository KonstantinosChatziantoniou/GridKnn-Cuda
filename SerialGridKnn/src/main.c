#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#include "../headers/pointsHelper.h"
#include "../headers/serialKnn.h"

void printTime(char* text, struct timeval end , struct timeval start){
    printf("%s ",text);
    long s=end.tv_sec-start.tv_sec;
    long us=end.tv_usec - start.tv_usec;
    if(us < 0){
        us = 1000000+us;
        s = s-1;
    }
    printf("%ld s, %ld us\n",s,us);
}
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
        seed = atoi(argv[4]);
    }
    int number_of_queries = number_of_points;
    const int dimensions = 3;
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
    
    float* knns = (float*) malloc(number_of_queries*dimensions*sizeof(int));
    float* knns_dists = (float*)malloc(number_of_queries*sizeof(float));

    for(int i = 0; i < number_of_queries; i++){
        knns_dists[i] = 100;
    }
    //printf("mallocs done\n");
    generatePoints(points, number_of_points, dimensions, 0, 1, 1);
    generatePoints(queries, number_of_queries, dimensions, 0, 1, 2);



    assignPointsToBlocks(points, block_of_point , points_per_block , side_block_length , number_of_points, grid_d , dimensions);
    assignPointsToBlocks(queries, block_of_query , queries_per_block , side_block_length , number_of_queries, grid_d , dimensions);

    for(int i = 0; i < grid_d*grid_d*grid_d; i++){
        integral_points_per_block[i] = 0;
        integral_queries_per_block[i] = 0;
        for(int j = 0; j < i; j++){
            integral_points_per_block[i] += points_per_block[j];
            integral_queries_per_block[i] += queries_per_block[j];
        }
    }

    
    rearrangePointsToGrid(points,grid_arranged_points, block_of_point , points_per_block , side_block_length , number_of_points, grid_d , dimensions);
    rearrangePointsToGrid(queries,grid_arranged_queries, block_of_query , queries_per_block , side_block_length , number_of_queries, grid_d , dimensions);

    assignPointsToBlocks(grid_arranged_points, block_of_point , points_per_block , side_block_length , number_of_points, grid_d , dimensions);
    assignPointsToBlocks(grid_arranged_queries, block_of_query , queries_per_block , side_block_length , number_of_queries, grid_d , dimensions);



    // printPointsToCsv("points.csv" , "w" , points , number_of_points , dimensions);
    // printPointsToCsv("queries.csv" , "w" , queries , number_of_queries , dimensions);
    free(points);
    free(queries);
    printf("binning done\n");

    gettimeofday(&tstart,NULL);
    float* ser_res_dists = (float*)malloc(number_of_queries*sizeof(float));
    int* ser_res_indx = (int*)malloc(number_of_queries*sizeof(int));
    for(int i = 0; i < number_of_queries; i++){
        ser_res_dists[i] = 100;
    }
    for(int b = 0; b < grid_d*grid_d*grid_d; b++){
        for(int q = 0; q < queries_per_block[b]; q++){
            int qrs_shifter = integral_queries_per_block[b] + q;
            //printf("query %d, shift %d\n",q,qrs_shifter);
            float nbrs_dist = 110;
            int nbrs_indx = 0;
            for(int p = 0; p < points_per_block[b]; p++){
                int pts_shifter = integral_points_per_block[b] + p;
               // printf("query %d, point %d, shift %d\n",q,p,pts_shifter);
                float dist = 0;
                for(int d = 0; d < 3; d++){
                    dist += pow(grid_arranged_queries[qrs_shifter*3 + d] - grid_arranged_points[pts_shifter*3 + d],2);
                }
                dist = sqrt(dist);


                if(dist < nbrs_dist){
                    //printf("dist %f, nbrsdist %f, index %d\n",dist,nbrs_dist,nbrs_indx);
                    nbrs_dist = dist;
                    nbrs_indx = pts_shifter;
                }
            }
            //printf("---------------------nbrs dist %f, ser dist %f\n",nbrs_dist,ser_res_dists[qrs_shifter]);
            if(nbrs_dist < ser_res_dists[qrs_shifter]){
                ser_res_dists[qrs_shifter] = nbrs_dist;
                ser_res_indx[qrs_shifter] = nbrs_indx;
                //printf("-------------------- qrs index %d, indx %d, dist %f\n",qrs_shifter,nbrs_indx,nbrs_dist);
            }
        }
    }


    gettimeofday(&tend,NULL);
    printTime("CPU KNN 1st PART ",tend,tstart);




     int counterOutCand = 0;
   //Knn NeighbourBlocks 
    for(int q = 0; q < number_of_queries; q++){
        float max_dist = ser_res_dists[q];
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
                int* currentBlock = (int*)malloc(dimensions*sizeof(int));
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
                float dist;
                for(int p = 0; p < numberOfPointsInBlock; p++){
                    dist = 0;
                    for(int d = 0; d < 3; d++){
                        dist += pow(grid_arranged_queries[q*3+d] - grid_arranged_points[(p+startingAddressOfPoints)*3 + d],2);
                    }
                    dist = sqrt(dist);


                    if(dist < ser_res_dists[q]){
                    ser_res_dists[q] = dist;
                    ser_res_indx[q] = p + startingAddressOfPoints;
                }
                }

                // if(q == 21721 || q == 26679){
                // printf("Nbs of query: ");
                // printPoints(&grid_arranged_queries[q*dimensions],1,dimensions);
                // printf("----Neighbours----\n");
                // printPoints(&grid_arranged_points[ser_res_indx[q]], k_num , dimensions);
                // printf("----Dists----\n");
                // printf("dist %.12f\n", ser_res_dists[q]);
                // printf("---------------------------\n");
                // }

                free(currentBlock);
            }

            free(nb.nbr_blocks);
        }
    }

    gettimeofday(&tend,NULL);
    printTime("CPU KNN 2nd PART ",tend,tstart);

     for(int i = 0; i < number_of_queries; i++){
        memcpy(&knns[i*3], &grid_arranged_points[ser_res_indx[i]*3], 3*sizeof(float));
    }
    printPointsToCsv("knn.csv" , "w" , knns , number_of_queries , dimensions);
    







    //Time measuring (end)//
    gettimeofday(&tend,NULL);
    printf("Time elapsed %ld s, %ld us\n",tend.tv_sec - tstart.tv_sec, tend.tv_usec - tstart.tv_usec);
    printf("%f %% searched other boxes (%d/%d)\n  ",100*(float)counterOutCand/number_of_queries,counterOutCand,number_of_queries);
    printPointsToCsv("knn.csv" , "w" , knns , number_of_queries , dimensions);
    printPointsToCsv("points_arranged.csv" ,"w" , grid_arranged_points , number_of_points , dimensions);
    printPointsToCsv("queries_arranged.csv" , "w" , grid_arranged_queries , number_of_queries , dimensions);





    // float* points = malloc(number_of_points*dimensions*sizeof(float));
    // float* queries = malloc(number_of_queries*dimensions*sizeof(float));
    // float* grid_arranged_points = malloc(number_of_points*dimensions*sizeof(float));
    // float* grid_arranged_queries = malloc(number_of_queries*dimensions*sizeof(float));
    // int* block_of_point = malloc(number_of_points*dimensions*sizeof(int));
    // int* block_of_query = malloc(number_of_queries*dimensions*sizeof(int));
    // int* points_per_block = malloc(grid_d*grid_d*grid_d*sizeof(int));
    // int* queries_per_block = malloc(grid_d*grid_d*grid_d*sizeof(int));
    // int* integral_points_per_block = malloc(grid_d*grid_d*grid_d*sizeof(int));
    // int* integral_queries_per_block = malloc(grid_d*grid_d*grid_d*sizeof(int));
    
    // float* knns = (float*) malloc(number_of_queries*dimensions*sizeof(int));
    // float* knns_dists = (float*)malloc(number_of_queries*sizeof(float));


    // float* ser_res_dists = (float*)malloc(number_of_queries*sizeof(float));
    // int* ser_res_indx = (int*)malloc(number_of_queries*sizeof(int));
    free(grid_arranged_points);
    free(grid_arranged_queries);
    free(block_of_point);
    free(block_of_query);
    free(points_per_block);
    free(queries_per_block);
    free(integral_points_per_block);
    free(integral_queries_per_block);
    free(knns);
    free(knns_dists);
    free(ser_res_dists);
    free(ser_res_indx);

    
    
    
    return 0;
}