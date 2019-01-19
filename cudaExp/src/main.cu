#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <math_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "../headers/serialKnn.h"
#include "../headers/pointsHelper.h"
void printPoints(int* pts, int num , int dim);

void printTime(char* text, struct timeval end , struct timeval start);


__global__ void devKnnShared(float* points, float* queries, int* points_per_block, int* queries_per_block, int* res_indexes , float* res_dists, int number_of_queries , int max_points)
{
    int b = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
    int num_of_queries = queries_per_block[b];
    int mulq = 0;
    int integral_queries = 0;
    int integral_points = 0;
    int qrs_shifter;
    int num_of_points ;
    float nbrs_dist;
    int nbrs_indx;
    int mulp ;
    int grid_d = gridDim.x;
    __shared__ float sh_pts[532][3];
    int flag = 0;
    float myQuery[3];
    for(int i = 0; i < b; i++){
        integral_points += points_per_block[i];
        integral_queries += queries_per_block[i];
    }
    while(mulq*blockDim.x < num_of_queries){
        int q = mulq*blockDim.x + threadIdx.x;
        qrs_shifter = integral_queries + q;
        num_of_points = points_per_block[b];
        nbrs_dist = 100;
        nbrs_indx = 1;
        mulp = 0;
        if(q < num_of_queries){
            myQuery[0] = queries[qrs_shifter*3 + 0];
            myQuery[1] = queries[qrs_shifter*3 + 1];
            myQuery[2] = queries[qrs_shifter*3 + 2];
        }
        while(mulp*blockDim.x < num_of_points){
            int p2 = mulp*blockDim.x + threadIdx.x;
            int pts_shifter2 = integral_points + p2;

            __syncthreads();
            if(p2 < num_of_points && pts_shifter2 < max_points){
                for(int d = 0; d < 3; d++){
                    sh_pts[threadIdx.x][d] = points[pts_shifter2*3 + d];
                }
            }
            __syncthreads();

            if(q < num_of_queries){
                int limit = min(num_of_points,(mulp+1)*blockDim.x);
                for(int p = mulp*blockDim.x; p < limit; p++){
                    int pts_shifter = integral_points + p;
                    float dist = 0;
                    for(int d = 0; d < 3; d++){
                        dist += powf(myQuery[d]- sh_pts[p - mulp*blockDim.x][d] ,2);   //points[pts_shifter*3+d],2); //   
                    }
                    dist = sqrtf(dist);
                    if(dist < nbrs_dist){
                        nbrs_dist = dist;
                        nbrs_indx = pts_shifter;
                    }  
                }
            }
            mulp++;
        }

        if(q < num_of_queries){
            if(nbrs_dist < res_dists[qrs_shifter]){
                res_dists[qrs_shifter] = nbrs_dist;
                res_indexes[qrs_shifter] = nbrs_indx;
            }
        }
               
        
       

        mulq++;
    }

    // Search neighbour blocks

    int nbrs_blocks[27];
    int number_of_nbrs_blocks = 0; 
    for(int i = -1; i <= 1; i++){
        for(int j = -1; j <= 1; j++){
            for(int k = -1; k <= 1; k++){
                if(i != 0 | j != 0 | k != 0){
                    int nx = blockIdx.x + i;
                    int ny = blockIdx.y  + j;
                    int nz = blockIdx.z  + k;

                    if(!(nx<0 | ny<0 | nz<0 | nx >=grid_d | ny >= grid_d | nz>=grid_d)){
                        nbrs_blocks[number_of_nbrs_blocks] = nx*grid_d*grid_d + ny*grid_d + nz;
                        number_of_nbrs_blocks++;
                    }
                }
            }
        }
    }
    for(int nb = 0; nb < number_of_nbrs_blocks; nb++){
        integral_points = 0;
        for(int i = 0; i < nbrs_blocks[nb]; i++){
            integral_points += points_per_block[i];
        }
        mulq = 0;
        while(mulq*blockDim.x < num_of_queries){
            int q = mulq*blockDim.x + threadIdx.x;
            qrs_shifter = integral_queries + q;
            num_of_points = points_per_block[nbrs_blocks[nb]];
            nbrs_dist = 100;
            nbrs_indx = 1;
            mulp = 0;
            if(q < num_of_queries){
                myQuery[0] = queries[qrs_shifter*3 + 0];
                myQuery[1] = queries[qrs_shifter*3 + 1];
                myQuery[2] = queries[qrs_shifter*3 + 2];
            }
            
            while(mulp*blockDim.x < num_of_points){
                int p2 = mulp*blockDim.x + threadIdx.x;
                int pts_shifter2 = integral_points + p2;

                __syncthreads();
                if(p2 < num_of_points && pts_shifter2 < max_points){
                    for(int d = 0; d < 3; d++){
                        sh_pts[threadIdx.x][d] = points[pts_shifter2*3 + d];
                    }
                }
                __syncthreads();

                if(q < num_of_queries){
                    int limit = min(num_of_points,(mulp+1)*blockDim.x);
                    for(int p = mulp*blockDim.x; p < limit; p++){
                        int pts_shifter = integral_points + p;
                        float dist = 0;
                        for(int d = 0; d < 3; d++){
                            dist += powf(myQuery[d]- sh_pts[p - mulp*blockDim.x][d] ,2);   //points[pts_shifter*3+d],2); //   
                        }
                        dist = sqrtf(dist);
                        if(dist < nbrs_dist){
                            nbrs_dist = dist;
                            nbrs_indx = pts_shifter;
                        }  
                    }
                }
                mulp++;
            }

            if(q < num_of_queries){
                if(nbrs_dist < res_dists[qrs_shifter]){
                    res_dists[qrs_shifter] = nbrs_dist;
                    res_indexes[qrs_shifter] = nbrs_indx;
                }
            }
                
            
        

            mulq++;
        }
    }
}

/******************** INPUT *********************
 * 1st param -> number of points    (default 2^5)
 * 2nd param -> grid dimensions     (default 2^1)
 * 3rd param -> seed                (default 1,2)
*************************************************/
int main(int argc, char** argv){

    cudaDeviceReset();
    struct timeval totalProgramStart,totalProgramEnd,tstart,tend;
    gettimeofday(&totalProgramStart,NULL);
    int *cudaInit;
    //----------------------------------------------//
    
    int number_of_points = 5;
    int grid_d = 1;
    int k_num = 1;
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
    gettimeofday(&tstart,NULL);
    float* points = (float*)malloc(number_of_points*dimensions*sizeof(float));
    float* queries = (float*)malloc(number_of_queries*dimensions*sizeof(float));
    float* grid_arranged_points = (float*)malloc(number_of_points*dimensions*sizeof(float));
    float* grid_arranged_queries = (float*)malloc(number_of_queries*dimensions*sizeof(float));
    int* block_of_point = (int*)malloc(number_of_points*dimensions*sizeof(int));
    int* block_of_query = (int*)malloc(number_of_queries*dimensions*sizeof(int));
    int* points_per_block = (int*)malloc(grid_d*grid_d*grid_d*sizeof(int));
    int* queries_per_block = (int*)malloc(grid_d*grid_d*grid_d*sizeof(int));
    int* integral_points_per_block = (int*)malloc(grid_d*grid_d*grid_d*sizeof(int));
    int* integral_queries_per_block = (int*)malloc(grid_d*grid_d*grid_d*sizeof(int));

    
    float* knns = (float*) malloc(number_of_queries*dimensions*sizeof(int));
    float* knns_gpu = (float*) malloc(number_of_queries*dimensions*sizeof(int));
    float* knns_dists = (float*)malloc(number_of_queries*sizeof(float));

    for(int i = 0; i < number_of_queries; i++){
        knns_dists[i] = 100;
    }

    gettimeofday(&tend,NULL);
    printTime("CPU MALLOC TIME ",tend,tstart);

    gettimeofday(&tstart,NULL);
    generatePoints(points, number_of_points, dimensions, 0, 1, 1);
    generatePoints(queries, number_of_queries, dimensions, 0, 1, 2);
    gettimeofday(&tend,NULL);
    printTime("GENERATION TIME ",tend,tstart);

    gettimeofday(&tstart,NULL);
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

    gettimeofday(&tend,NULL);
    printTime("CPU BINNING TIME ",tend,tstart);

    
  gettimeofday(&tstart,NULL);
    float* dev_points;
    cudaError_t cuer;
    cuer = cudaMalloc(&dev_points,number_of_points*3*sizeof(float));
    printf("%s\n",cudaGetErrorName(cuer));
    cuer = cudaMemcpy(dev_points, grid_arranged_points, number_of_points*3*sizeof(float),cudaMemcpyHostToDevice);
    printf("%s\n",cudaGetErrorName(cuer));
    float* dev_queries;
    cuer = cudaMalloc(&dev_queries, number_of_queries*3*sizeof(float));
    printf("%s\n",cudaGetErrorName(cuer));
    cuer = cudaMemcpy(dev_queries,grid_arranged_queries,number_of_queries*3*sizeof(float),cudaMemcpyHostToDevice);
    printf("%s\n",cudaGetErrorName(cuer));
    int* dev_points_per_block;
    cuer = cudaMalloc(&dev_points_per_block, grid_d*grid_d*grid_d*sizeof(int));
    printf("%s\n",cudaGetErrorName(cuer));
    cuer = cudaMemcpy(dev_points_per_block , points_per_block, grid_d*grid_d*grid_d*sizeof(int),cudaMemcpyHostToDevice);
    printf("%s\n",cudaGetErrorName(cuer));
    int* dev_queries_per_blcok;
    cuer = cudaMalloc(&dev_queries_per_blcok, grid_d*grid_d*grid_d*sizeof(int));
    printf("%s\n",cudaGetErrorName(cuer));
    cuer = cudaMemcpy(dev_queries_per_blcok, queries_per_block ,grid_d*grid_d*grid_d*sizeof(int),cudaMemcpyHostToDevice);
    printf("%s\n",cudaGetErrorName(cuer));



    float* res_dists = (float*)malloc(number_of_queries*sizeof(float));
    int* res_indexes = (int*)malloc(number_of_queries*sizeof(int));

    float* res_dists2 = (float*)malloc(number_of_queries*sizeof(float));
    int* res_indexes2 = (int*)malloc(number_of_queries*sizeof(int));
    for(int i = 0; i < number_of_queries; i++){
        res_dists[i] = 100;
        res_indexes[i] = 19;
    }
    int* dev_res_indexes;
    cuer = cudaMalloc(&dev_res_indexes,number_of_queries*sizeof(int));
    printf("%s\n",cudaGetErrorName(cuer));
    cuer = cudaMemcpy(dev_res_indexes,res_indexes, number_of_queries*sizeof(int),cudaMemcpyHostToDevice);
    printf("%s\n",cudaGetErrorName(cuer));
    
    float *dev_res_dists;
    cuer = cudaMalloc(&dev_res_dists,number_of_queries*sizeof(float));
    printf("%s\n",cudaGetErrorName(cuer));
    cuer = cudaMemcpy(dev_res_dists,res_dists,number_of_queries*sizeof(float) , cudaMemcpyHostToDevice);
    printf("%s\n",cudaGetErrorName(cuer));

    gettimeofday(&tend,NULL);
    printTime("GPU MALLOC",tend,tstart);

    

    //dbgKnn<<<1000,500>>>(dev_res_dists,dev_res_indexes,number_of_queries);
    devKnnShared<<<dim3(grid_d,grid_d,grid_d),512>>>(dev_points,dev_queries,dev_points_per_block , dev_queries_per_blcok,dev_res_indexes , dev_res_dists , number_of_queries,number_of_points);
    cuer = cudaGetLastError();
    printf("%s\n",cudaGetErrorName(cuer));
    cuer = cudaMemcpy(res_dists,dev_res_dists,number_of_queries*sizeof(float) , cudaMemcpyDeviceToHost);
    printf("%s\n",cudaGetErrorName(cuer));
    cuer = cudaMemcpy(res_indexes,dev_res_indexes, number_of_queries*sizeof(int),cudaMemcpyDeviceToHost);
    printf("%s\n",cudaGetErrorName(cuer));


         for(int i = 0; i < number_of_queries; i++){
        memcpy(&knns_gpu[i*3], &grid_arranged_points[res_indexes[i]*3], 3*sizeof(float));
    }
    printPointsToCsv("knn2.csv" , "w" , knns_gpu , number_of_queries , dimensions);
    gettimeofday(&tstart,NULL);
    printTime("GPU KNN ",tstart,tend);
   


    printPointsToCsv("knn.csv" , "w" , knns_gpu , number_of_queries*k_num , dimensions);
    printPointsToCsv("points.csv" , "w" , points , number_of_points , dimensions);
    printPointsToCsv("queries.csv" , "w" , queries , number_of_queries , dimensions);
    printPointsToCsv("points_arranged.csv" ,"w" , grid_arranged_points , number_of_points , dimensions);
    printPointsToCsv("queries_arranged.csv" , "w" , grid_arranged_queries , number_of_queries , dimensions);

    






    //debugGPUKnnGlobal(grid_arranged_points,0,0,grid_arranged_queries,0,0,points_per_block,queries_per_block,grid_d,num_of_threads,1,indxs,dsts);














    free(points);
    free(queries);
    free(grid_arranged_points);
    free(grid_arranged_queries);
    free(block_of_point);
    free(block_of_query);
    free(points_per_block);
    free(queries_per_block);
    //free(knn_res);
    //----------------------------------------------------------------//
    gettimeofday(&totalProgramEnd,NULL);
    printTime("total program time ", totalProgramEnd,totalProgramStart);
    cudaProfilerStop();
    cudaDeviceReset();
    return 0;
}






void printPoints(int* pts, int num, int dim){
    for(int i = 0; i < num; i++){
        printf("Points%d:\t",i);
        for(int j = 0; j < dim; j++){
            printf("x%d %d\t",j,pts[i*dim + j]);
        }
        printf("\n");
    }
}


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
