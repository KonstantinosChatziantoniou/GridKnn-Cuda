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

__global__
void GPUfindDistancesFromQuery(float* points, int pointIndex , float *queries , int queryIndex ,float* dists, int numberOfPoints , int dimensions);
  
__global__
void kernel(int* pts, int* queries , int qIndex,int* res, int num_pts, int dimensions);

__global__
void GPUassignePointsToBlocks(float* dev_points, int* dev_block_of_pts,float side_block_length,int number_of_points,int dimensions);

/******************** INPUT *********************
 * 1st param -> number of points    (default 2^5)
 * 2nd param -> grid dimensions     (default 2^1)
 * 3rd param -> seed                (default 1,2)
*************************************************/
void getDeviceInfo(){
    cudaDeviceProp cp;
    cudaGetDeviceProperties(&cp,0);
    printf("%s %d %d\n",cp.name,cp.major,cp.minor);
    printf("Global Memory: %lu\n", cp.totalGlobalMem);
}
int main(int argc, char** argv){
    getDeviceInfo();
    cudaDeviceReset();
    struct timeval totalProgramStart,totalProgramEnd,tstart,tend;
    gettimeofday(&totalProgramStart,NULL);
    int *cudaInit;
    //----------------------------------------------//
    
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

    

    kNeighbours* knn_struct = (kNeighbours*)malloc(number_of_queries*sizeof(kNeighbours));
    for(int q = 0; q < number_of_queries; q++){
        knn_struct[q].nb_points = (float*)malloc(k_num*dimensions*sizeof(float));
        knn_struct[q].dists = (float*)malloc(k_num*sizeof(float));
        knn_struct[q].max_dist = 100;
        knn_struct[q].max_index = -1;
        knn_struct[q].num_of_nbrs = 0;
        knn_struct[q].k = k_num;
    }

    //--------------------------CUDA POINTERS------------------------//
    float *dev_pts,*dev_qrs;
    //int *dev_block_of_pts,*dev_block_of_qrs;

    gettimeofday(&tstart,NULL);

    cudaMalloc(&dev_pts,number_of_points*dimensions*sizeof(float));
    cudaMalloc(&dev_qrs,number_of_queries*dimensions*sizeof(float));
    //cudaMalloc(&dev_block_of_pts, number_of_points*dimensions*sizeof(float));
    //cudaMalloc(&dev_block_of_qrs, number_of_queries*dimensions*sizeof(float));

    gettimeofday(&tend,NULL);
    printTime("Cuda Mallocs ",tend,tstart);

    gettimeofday(&tstart,NULL);
    generatePoints(points, number_of_points, dimensions, 0, 1, 1);
    generatePoints(queries, number_of_queries, dimensions, 0, 1, 2);
    gettimeofday(&tend,NULL);
    printTime("Gen time ",tend,tstart);

    

    /*
    ////printPointsToCsv("points.csv","w",points,number_of_points,dimensions);
    int tmp_blocks = number_of_points/128;
    //dim3 gridsDim(tmp_blocks);
    //dim3 blocksDim(128,3);

    gettimeofday(&tstart,NULL);
    cudaMemcpy(dev_pts,points,number_of_points*dimensions*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_qrs,queries,number_of_points*dimensions*sizeof(float),cudaMemcpyHostToDevice);
    cudaError ce;
    ce = cudaGetLastError();
    printf("%s",cudaGetErrorName(ce));
    //void GPUassignePointsToBlocks(float* dev_points, int* dev_block_of_pts,float side_block_length,int number_of_points,int dimensions){
    GPUassignePointsToBlocks<<<tmp_blocks,128>>>(dev_pts,dev_block_of_pts,side_block_length,number_of_points,dimensions);
    GPUassignePointsToBlocks<<<tmp_blocks,128>>>(dev_qrs,dev_block_of_qrs,side_block_length,number_of_points,dimensions);
    
    //cudaDeviceSynchronize();
    cudaMemcpy(block_of_point,dev_block_of_pts,number_of_points*dimensions*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_of_query,dev_block_of_qrs,number_of_points*dimensions*sizeof(int), cudaMemcpyDeviceToHost);
    gettimeofday(&tend,NULL);
    printTime("GPU assignement time ",tend,tstart);



    //---------CPU serial ---------------//
    gettimeofday(&tstart,NULL);
    assignPointsToBlocks(points,block_of_point,points_per_block,side_block_length,number_of_points,grid_d,dimensions);
    assignPointsToBlocks(queries,block_of_query,points_per_block,side_block_length,number_of_points,grid_d,dimensions);
    gettimeofday(&tend,NULL);
    printTime("CPU assignement time ",tend,tstart);

    //----------Same but async----------//

    gettimeofday(&tstart,NULL);
    cudaStream_t custrm[2];
    cudaStreamCreate(&custrm[0]);
    cudaStreamCreate(&custrm[1]);
    cudaMemcpyAsync(dev_pts,points,number_of_points*dimensions*sizeof(float),cudaMemcpyHostToDevice,custrm[0]);
    cudaMemcpyAsync(dev_qrs,queries,number_of_points*dimensions*sizeof(float),cudaMemcpyHostToDevice,custrm[1]);

    //void GPUassignePointsToBlocks(float* dev_points, int* dev_block_of_pts,float side_block_length,int number_of_points,int dimensions){
    GPUassignePointsToBlocks<<<tmp_blocks,128,0,custrm[0]>>>(dev_pts,dev_block_of_pts,side_block_length,number_of_points,dimensions);
    GPUassignePointsToBlocks<<<tmp_blocks,128,0,custrm[1]>>>(dev_qrs,dev_block_of_qrs,side_block_length,number_of_points,dimensions);
    
    //cudaDeviceSynchronize();
    cudaMemcpyAsync(block_of_point,dev_block_of_pts,number_of_points*dimensions*sizeof(int), cudaMemcpyDeviceToHost,custrm[0]);
    cudaMemcpyAsync(block_of_query,dev_block_of_qrs,number_of_points*dimensions*sizeof(int), cudaMemcpyDeviceToHost,custrm[1]);

    cudaStreamSynchronize(custrm[0]);

    cudaStreamSynchronize(custrm[1]);
    gettimeofday(&tend,NULL);
    printTime("GPU async assignement time ",tend,tstart);

    cudaStreamDestroy(custrm[0]);
    cudaStreamDestroy(custrm[1]);






    for(int i = 0; i < 100; i++){
        for(int j = 0; j < dimensions; j++){
            printf("cuda vs cpu [%d %d]\t",block_of_query[i*dimensions+j],block_of_point[i*dimensions+j]);
        }
        printf("\n");
    }
   */




    //---------CPU serial ---------------//
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            gettimeofday(&tstart,NULL);
            assignPointsToBlocks(points,block_of_point,points_per_block,side_block_length,number_of_points,grid_d,dimensions);
            assignPointsToBlocks(queries,block_of_query,queries_per_block,side_block_length,number_of_queries,grid_d,dimensions);
            

            rearrangePointsToGrid(points,grid_arranged_points, block_of_point , points_per_block , side_block_length , number_of_points, grid_d , dimensions);
            rearrangePointsToGrid(queries,grid_arranged_queries, block_of_query , queries_per_block , side_block_length , number_of_queries, grid_d , dimensions);

            assignPointsToBlocks(grid_arranged_points, block_of_point , points_per_block , side_block_length , number_of_points, grid_d , dimensions);
            assignPointsToBlocks(grid_arranged_queries, block_of_query , queries_per_block , side_block_length , number_of_queries, grid_d , dimensions);
            for(int i = 0; i < grid_d*grid_d*grid_d; i++){
                integral_points_per_block[i] = 0;
                integral_queries_per_block[i] = 0;
                for(int j = 0; j < i; j++){
                    integral_points_per_block[i] += points_per_block[j];
                    integral_queries_per_block[i] += queries_per_block[j];
                }
            }

            gettimeofday(&tend,NULL);
            printTime("CPU assignement time ",tend,tstart);
        }

        #pragma omp section
        {
            cudaMemcpy(dev_pts,grid_arranged_points,number_of_points*dimensions*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemcpy(dev_qrs,grid_arranged_queries,number_of_queries*dimensions*sizeof(float),cudaMemcpyHostToDevice);
        }


    }
    


    
    int max_points_pb  = -1;
    int max_queries_pb = -1;
    for(int i = 0; i < grid_d*grid_d*grid_d; i++){
        if(points_per_block[i] > max_points_pb){
            max_points_pb = points_per_block[i];
        }
        if(queries_per_block[i] > max_queries_pb){
            max_queries_pb = queries_per_block[i];
        }
    }

    float *dev_distances;
    float *kdists = (float*)malloc(max_points_pb*sizeof(float));
    cudaMalloc(&dev_distances,max_points_pb*sizeof(float));
/*
    //--Serial CPU -> GPU --//
    for(int q = 0; q < number_of_queries; q++){
        int* currentBlock = (int*)malloc(dimensions*sizeof(int));
        for(int i = 0; i < dimensions; i++){
            currentBlock[i] = block_of_query[q*dimensions + i];
        }
        int gridIndex = 0;
        for(int i = 0; i < dimensions; i++){
            gridIndex += pow(grid_d,dimensions -1 -i)*currentBlock[i];
        }
        int startingAddressOfPoints = integral_points_per_block[gridIndex];
        int numberOfPointsInBlock = points_per_block[gridIndex];
        int blocks = ceil((float)numberOfPointsInBlock/(float)16);
        //void GPUfindDistancesFromQuery(float* points, int pointIndex , float *queries , int queryIndex ,float* dists, int numberOfPoints , int dimensions){
        //printf("BLOOOOOOOOCKS %d x %d -> %d\n",blocks,numberOfPointsInBlock, blocks*16);            
        GPUfindDistancesFromQuery<<<blocks,16>>>(dev_pts,startingAddressOfPoints,dev_qrs,q,dev_distances,numberOfPointsInBlock,dimensions);
        
        cudaMemcpy(kdists,dev_distances,numberOfPointsInBlock*sizeof(float),cudaMemcpyDeviceToHost);
        //printPoints(kdists,numberOfPointsInBlock,1);
        //printf("for q%d - %d - %d\n",q,numberOfPointsInBlock,max_points_pb);
        for(int i = 0; i < numberOfPointsInBlock; i++){
            //printf("adding neighbour %d of %d - %f\n",i,startingAddressOfPoints*dimensions + i*dimensions,knn_struct[q].max_dist);
            addNeighbour(&knn_struct[q],&grid_arranged_points[startingAddressOfPoints*dimensions + i*dimensions],kdists[i],dimensions);
            //printf("added neighbour %d of %d\n",i,q);
            //float dbg_dist = 0;
            // for(int d = 0; d < dimensions; d++){
            //     dbg_dist += pow(grid_arranged_queries[q*dimensions+d] - grid_arranged_points[startingAddressOfPoints*dimensions + i*dimensions + d],2);
            // }
            // dbg_dist = sqrt(dbg_dist);
            //printPoints(&grid_arranged_points[startingAddressOfPoints*dimensions + i*dimensions],1,3);
            //printf("CPU dist %f , GPU dist %f\n",dbg_dist,kdists[i]);
        }            
        // printf("Nbs of query: ");
        //         printPoints(&grid_arranged_queries[q*dimensions],1,dimensions);
        //         printf("----Neighbours----\n");
        //         printPoints(knn_struct[q].nb_points, k_num , dimensions);
        //         printf("----Dists----\n");
        //         printPoints(knn_struct[q].dists,k_num,1);
        //         printf("---------------------------\n");
    }

    printf("malloced\n");
    */



    
    int number_of_threads = 4;
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(number_of_threads);
    float **p_dists = (float**)malloc(number_of_threads*sizeof(float*));
    float **dev_p_dists = (float**)malloc(number_of_threads*sizeof(float*));
    //cudaStream_t *custreams = (cudaStream_t*)malloc(number_of_threads)
    for(int i = 0; i < number_of_threads; i++){
        p_dists[i] = (float*)malloc(max_points_pb*sizeof(float));
        cudaMalloc(&dev_p_dists[i],max_points_pb*sizeof(float));
    }
    #pragma omp parallel for num_threads(number_of_threads)
        for(int q = 0; q < number_of_queries; q++){
            int pid = omp_get_thread_num();
            //printf("Thread %d for query %d// total threads =  %d\n",pid,q,omp_get_num_threads());
            int* currentBlock = (int*)malloc(dimensions*sizeof(int));
            for(int i = 0; i < dimensions; i++){
                currentBlock[i] = block_of_query[q*dimensions + i];
            }
            int gridIndex = 0;
            for(int i = 0; i < dimensions; i++){
                gridIndex += pow(grid_d,dimensions -1 -i)*currentBlock[i];
            }
            int startingAddressOfPoints = integral_points_per_block[gridIndex];
            int numberOfPointsInBlock = points_per_block[gridIndex];
            int blocks = ceil((float)numberOfPointsInBlock/(float)16);
            //void GPUfindDistancesFromQuery(float* points, int pointIndex , float *queries , int queryIndex ,float* dists, int numberOfPoints , int dimensions){
            //printf("BLOOOOOOOOCKS %d x %d -> %d\n",blocks,numberOfPointsInBlock, blocks*16);            
            GPUfindDistancesFromQuery<<<blocks,16>>>(dev_pts,startingAddressOfPoints,dev_qrs,q,dev_p_dists[pid],numberOfPointsInBlock,dimensions);
            
            cudaMemcpy(p_dists[pid],dev_p_dists[pid],numberOfPointsInBlock*sizeof(float),cudaMemcpyDeviceToHost);
            //printPoints(kdists,numberOfPointsInBlock,1);
            //printf("for q%d - %d - %d\n",q,numberOfPointsInBlock,max_points_pb);
            for(int i = 0; i < numberOfPointsInBlock; i++){
                //printf("adding neighbour %d of %d - %f\n",i,startingAddressOfPoints*dimensions + i*dimensions,knn_struct[q].max_dist);
                addNeighbour(&knn_struct[q],&grid_arranged_points[startingAddressOfPoints*dimensions + i*dimensions],p_dists[pid][i],dimensions);
                //printf("added neighbour %d of %d\n",i,q);
                //float dbg_dist = 0;
                // for(int d = 0; d < dimensions; d++){
                //     dbg_dist += pow(grid_arranged_queries[q*dimensions+d] - grid_arranged_points[startingAddressOfPoints*dimensions + i*dimensions + d],2);
                // }
                // dbg_dist = sqrt(dbg_dist);
                //printPoints(&grid_arranged_points[startingAddressOfPoints*dimensions + i*dimensions],1,3);
                //printf("CPU dist %f , GPU dist %f\n",dbg_dist,kdists[i]);
            }            
            // printf("Nbs of query: ");
            //         printPoints(&grid_arranged_queries[q*dimensions],1,dimensions);
            //         printf("----Neighbours----\n");
            //         printPoints(knn_struct[q].nb_points, k_num , dimensions);
            //         printf("----Dists----\n");
            //         printPoints(knn_struct[q].dists,k_num,1);
            //         printf("---------------------------\n");
            free(currentBlock);
        }   
    















    
    int counterOutCand = 0;
    // Knn NeighbourBlocks 
    #pragma omp parallel for num_threads(number_of_threads)
    for(int q = 0; q < number_of_queries; q++){
        int pid = omp_get_thread_num();
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
                int blocks = ceil((float)numberOfPointsInBlock/(float)16);
                //void GPUfindDistancesFromQuery(float* points, int pointIndex , float *queries , int queryIndex ,float* dists, int numberOfPoints , int dimensions){
                GPUfindDistancesFromQuery<<<blocks,16>>>(dev_pts,startingAddressOfPoints,dev_qrs,q,dev_p_dists[pid],numberOfPointsInBlock,dimensions);
                cudaMemcpy(p_dists[pid],dev_p_dists[pid],numberOfPointsInBlock*sizeof(float),cudaMemcpyDeviceToHost);
                
                for(int i = 0; i < numberOfPointsInBlock; i++){
                    addNeighbour(&knn_struct[q],&grid_arranged_points[startingAddressOfPoints*dimensions + i*dimensions],p_dists[pid][i],dimensions);
                }            
                //printf("Nbs of query: ");
                //printPoints(&grid_arranged_queries[q*dimensions],1,dimensions);
                //printf("----Neighbours----\n");
                //printPoints(knn_struct[q].nb_points, k_num , dimensions);
                //printf("---------------------------\n");
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

    


    float* knn_res = (float*)malloc(number_of_queries*k_num*dimensions*sizeof(float));
    for(int q = 0; q < number_of_queries; q++){
        //printPoints(knn_struct[q].nb_points, k_num , dimensions);
        memcpy(&knn_res[q*dimensions*k_num] , knn_struct[q].nb_points , k_num*dimensions*sizeof(float));
    }

    
    // float* dbgpoints = (float*)malloc(100*sizeof(float));
    // for(int i = 0; i < 10; i++){
    //     float dist = 0;
    //     for(int j = 0; j < 3; j++){
    //         dist += pow(grid_arranged_points[j] - grid_arranged_points[i*dimensions+j],2);
    //     }
    //     dbgpoints[i] = sqrtf(dist);
    // }
    
    // printPoints(dbgpoints,10,1);

    // GPUfindDistancesFromQuery<<<6,2>>>(dev_pts,0,dev_pts,0,dev_distances,10,dimensions);
    // cudaMemcpy(dbgpoints,dev_distances,10*sizeof(float),cudaMemcpyDeviceToHost);

    // printPoints(dbgpoints,10,1);

    printPointsToCsv("knn.csv" , "w" , knn_res , number_of_queries*k_num , dimensions);
    //printPointsToCsv("points.csv" , "w" , points , number_of_points , dimensions);
    //printPointsToCsv("queries.csv" , "w" , queries , number_of_queries , dimensions);
    printPointsToCsv("points_arranged.csv" ,"w" , grid_arranged_points , number_of_points , dimensions);
    printPointsToCsv("queries_arranged.csv" , "w" , grid_arranged_queries , number_of_queries , dimensions);



    int dbgcntr = 0;
    cudaFree(dev_pts);
    cudaFree(dev_qrs);
    cudaFree(dev_distances);
    for(int i = 0; i < number_of_threads; i++){
        free(p_dists[i]);
        cudaFree(dev_p_dists[i]);
    }
    free(p_dists);
    free(dev_p_dists);
   // cudaFree(dev_block_of_pts);
    //cudaFree(dev_block_of_qrs);

    free(points);
    free(queries);
    free(grid_arranged_points);
    free(grid_arranged_queries);
    free(block_of_point);
    free(block_of_query);
    free(points_per_block);
    free(queries_per_block);
    free(knn_res);

    for(int q = 0; q < number_of_queries; q++){
        free(knn_struct[q].nb_points);
        free(knn_struct[q].dists);
    }
    free(knn_struct);
    //free(knn_res);
    //----------------------------------------------------------------//
    gettimeofday(&totalProgramEnd,NULL);
    printTime("total program time ", totalProgramEnd,totalProgramStart);
    cudaProfilerStop();
    cudaDeviceReset();
    return 0;
}

__global__
void GPUfindDistancesFromQuery(float* points, int pointIndex , float *queries , int queryIndex ,float* dists, int numberOfPoints , int dimensions){
    int ix =  blockIdx.x*blockDim.x + threadIdx.x;
    if(ix < numberOfPoints){
        float dist = 0;
        for(int j = 0; j < dimensions; j++){
            dist += powf(queries[queryIndex*dimensions+j] - points[pointIndex*dimensions + ix*dimensions + j],2);
        }
        dists[ix] = sqrtf(dist);
    }
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
__global__
void GPUassignePointsToBlocks(float* dev_points, int* dev_block_of_pts,float side_block_length,int number_of_points,int dimensions){
    int ix =  blockIdx.x*blockDim.x + threadIdx.x;
    
    if(ix < number_of_points){
        
        for(int j = 0; j < dimensions; j++){
            //dev_block_of_pts[ix*dimensions + j] = dev_points[ix*dimensions + j]/side_block_length;
            dev_block_of_pts[ix*dimensions + j] =  __float2int_rd(dev_points[ix*dimensions + j]/side_block_length);
        }
    }
}

__global__
void kernel(int* pts, int* queries , int qIndex,int* res, int num_pts, int dimensions){
    int ix =  blockIdx.x*blockDim.x + threadIdx.x;
    
    if(ix < num_pts){
        int dist = 0;
        for(int i = 0; i < dimensions; i++){
            dist += abs(pts[ix*dimensions + i] - queries[qIndex*dimensions + i]);
        }
        res[ix] = dist;
    }

}