/*
MP-aGMRES is a collection of GPU-accelerated GMRES implementations of Restarted GMRES that utilize mixed precision and/or varied restarts.

Copyright Notice:
    Copyright 2025 Abir Haque

License Notice:
    This file is part of MP-aGMRES

    MP-aGMRES is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License Version 3 as published by
    the Free Software Foundation.

    MP-aGMRES is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License Version 3 for more details.

    You should have received a copy of the GNU Affero General Public License Version 3
    along with MP-aGMRES in the file labeled LICENSE.txt.  If not, see https://www.gnu.org/licenses/agpl-3.0.txt


Author:
    Abir Haque

Date Last Updated:
    September 6th, 2025

Notes:
    This software was developed by Abir Haque in collaboration with Dr. Suzanne M. Shontz and Dr. Xuemin Tu at the University of Kansas (KU).
    This work was supported by the following:
        HPC facilities operated by the Center for Research Computing at KU supported by NSF Grant OAC-2117449,
        REU Supplement to NSF Grant OAC-1808553,
        REU Supplement to NSF Grant CBET-2245153,
        KU School of Engineering Undergraduate Research Fellows Program
    If you wish to use this code in your own work, you must review the license at LICENSE.txt and cite the following paper:
        Abir Haque, Suzanne Shontz, Xuemin Tu. GPU-Accelerated, Mixed Precision GMRES(m) with Varied Restarts. 2025 IEEE High Performance Extreme Computing Conference (HPEC), September 2025
        Paper Link: TBD
*/
#include <matio.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <chrono>
using namespace std;
using namespace std::chrono;
using namespace Eigen;


__global__ void warmup(){
  unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
  float a,b;
  a=b=0.0f;
  b+=a+tid; 
}


#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                       \
{                                                                              \
    cublasStatus_t status = (func);                                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cublasGetStatusString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

__global__ void put_vec64_div_scalar64_to_vec64(
    double* out,
    double* x,
    double scalar,
    int N
)
{  
    int tid = blockDim.x * blockIdx.x + threadIdx.x ;
    if(tid<N){
        out[tid]=x[tid]/scalar;
    }
}

__global__ void rescale_Q_Tjp1_if_needed(
    double* hjp1,
    double tol,
    double* out,
    int N,
    int j
)
{  
    if(fabs(hjp1[j])>tol){
        int tid = blockDim.x * blockIdx.x + threadIdx.x ;
        if(tid<N){
            out[tid]/=hjp1[j];
        }
    }
}

__global__ void mv(
    double* hi,
    int* j,
    double* val
)
{  
    hi[j[0]]=val[0];
}

__global__ void negate_val(
    double* in,
    int j,
    double* out
)
{  
    int tid = blockDim.x * blockIdx.x + threadIdx.x ;
    if(tid==0){
        out[j]=-1.0*in[0];
    }
}

__global__ void updateH1(
    double* hi,
    double* hip1,
    double& a_i,
    double& b_i,
    int j
)
{  
    int tid = blockDim.x * blockIdx.x + threadIdx.x ;
    if(tid==0){
        double tmp1=a_i*hi[j]+b_i*hip1[j];
        hip1[j] = -b_i*hi[j]+a_i*hip1[j];
        hi[j]=tmp1;
    }
}
__global__ void last_updates(
    double* hj,
    double* hjp1,
    double* a_,
    double* b_,
    double* s_,
    int j,
    double* sjp1
)
{  
    //givens coeffs
    int tid = blockDim.x * blockIdx.x + threadIdx.x ;
    if(tid==0){
        double c,s;
        double tmp1,tmp2;
        double tmp;
        if(abs(hjp1[j])<0.000001){
            c=1.0;
            s=0.0;
        }
        else if(abs(hjp1[j])>abs(hj[j])){
            tmp=hj[j]/hjp1[j];
            s=1.0/sqrt(1.0+tmp*tmp);
            c=tmp*s;
        }
        else{
            tmp=hjp1[j]/hj[j];
            c=1.0/sqrt(1.0+tmp*tmp);
            s=tmp*c;
        }
        a_[j]=c;
        b_[j]=s;
        tmp1=a_[j]*s_[j];
        s_[j+1]=-b_[j]*s_[j];
        s_[j]=tmp1;
        hj[j]=a_[j]*hj[j]+b_[j]*hjp1[j];
        hjp1[j]=0.0;
        sjp1[0]=abs(b_[j]*s_[j]);
    }
}

__global__ void div_val(
    double* y,
    double* Hi,
    int* i
)
{
    y[i[0]]/=Hi[i[0]];
}

__global__ void final_xi_update(
    int* i,
    double* x,
    double* Q_Tl,
    double* y,
    int* l

)
{
    x[i[0]]+=(double)Q_Tl[i[0]]*(double)y[l[0]];
}



__global__ void vec_sub_scalar_vec(
    double* x,
    double* y,
    double* scalar,
    int* N
)
{  
    int tid = blockDim.x * blockIdx.x + threadIdx.x ;
    int i,j,k;
    if(tid<N[0]){
        x[tid]-=scalar[0]*y[tid];
    }
}
int dynamic_restarted_gmres(
    int* A_outerIndexPtr,
    int* A_innerIndexPtr,
    double* A_valuePtr_double,
    double* A_valuePtr_float,
    int A_num_outerIndexPtr,
    int A_num_innerIndexPtr,
    int A_num_valuePtr,
    double* b,
    double* x0,
    int N,
    int k,
    int max_restarts,
    double tol,
    double* x,
    int& iters,
    int& restarts,
    bool& converged,
    double& res,
    float& gpu_time
)
{



    double beta,b_norm;
    double b_norm_double;
    double prev_err;
    double cur_err;
    double r0_norm;
    double cr = 1;
    double prev_m = k;
    double cur_m;
    double tmp;
    double tmp_float;
    int orig_k=k;
    int d=(k*10)/100;
    int total_iters=0;
    double max_cr=cos((8*M_PI)/180);
    double min_cr=cos((80*M_PI)/180);
    double* xi=(double*) malloc(N*sizeof(double));
    double* ri=(double*) malloc(N*sizeof(double));
    memcpy(xi,x0,N*sizeof(double));
    double* tmp_vec=(double*) calloc(N,sizeof(double));
    double* tmp_float_vec=(double*) calloc(N,sizeof(double));

    int i,j,l,m,n;
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);


    int* gpu_A_outerIndexPtr;
    int* gpu_A_innerIndexPtr;
    double* gpu_A_valuePtr_float;
    double* gpu_A_valuePtr_double;
    int* gpu_A_num_outerIndexPtr;
    int* gpu_A_num_innerIndexPtr;
    int* gpu_A_num_valuePtr;
    int* gpu_N;
    int* gpu_i;
    int* gpu_j;
    double* gpu_nrm_double;
    double* gpu_val;
    double* gpu_xi_double;
    double* gpu_ri_double;
    double* gpu_Axi_double;
    double* gpu_tmp_double;
    double* gpu_b_double;
    double* gpu_y;
    double* gpu_z;
    double neg_one_float = -1.0;
    double one_float = 1.0;
    double zero_float = 0.0;
    double one_double = 1.0;
    double zero_double = 0.0;
    double neg_one_double = -1.0;
    double* gpu_neg_one_float;
    double* gpu_one_float;
    double* gpu_zero_float;
    double* gpu_one_double;
    double* gpu_zero_double;
    double* gpu_neg_one_double;
    size_t bufferSize=0;
    void* externalBuffer;
    double* s ;
    double* a_;
    double* b_;
    double* gpu_sjp1;

    double* y=(double*) calloc(N,sizeof(double));


    double* Q_T[k+1];
    double* H[k+1];

    double* Q_T_local[k+1]; //Transpose Q since we only access colums of Q for most of GMRES
    for(i=0;i<k+1;i++){
        Q_T_local[i]=(double*)calloc(N,sizeof(double));
    } 
    double H_local[k+1][k]={0}; 
    double s_local[k+1]={0};

    gpu_time=0.0;
    float tmp_gpu_time=0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublasHandle_t handle;
    cublasCreate(&handle);
    CHECK_CUDA(cudaMalloc((void**)&gpu_neg_one_float , sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_one_float     , sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_zero_float    , sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_one_double    , sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_zero_double   , sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_neg_one_double, sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_sjp1, sizeof(double)));
    CHECK_CUDA(cudaMemset(gpu_neg_one_float , -1.0, sizeof(double)));
    CHECK_CUDA(cudaMemset(gpu_one_float     , 1.0, sizeof(double)));
    CHECK_CUDA(cudaMemset(gpu_zero_float    , 0.0, sizeof(double)));
    CHECK_CUDA(cudaMemset(gpu_one_double    , 1.0, sizeof(double)));
    CHECK_CUDA(cudaMemset(gpu_zero_double   , 0.0, sizeof(double)));
    CHECK_CUDA(cudaMemset(gpu_neg_one_double, -1.0, sizeof(double)));
    CHECK_CUDA(cudaMemset(gpu_sjp1, 0.0, sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_A_outerIndexPtr, A_num_outerIndexPtr*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_A_innerIndexPtr, A_num_innerIndexPtr*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_A_valuePtr_float, A_num_valuePtr*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_A_valuePtr_double, A_num_valuePtr*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_A_num_outerIndexPtr, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_A_num_innerIndexPtr, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_A_num_valuePtr, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_N, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_i, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_j, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_nrm_double, sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_val, sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_xi_double, N*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_b_double, N*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_Axi_double, N*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_tmp_double, N*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_ri_double, N*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_y, N*sizeof(double)));
    CHECK_CUDA(cudaMemset(gpu_y, 0, N*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&gpu_z, N*sizeof(double)));

    CHECK_CUDA(cudaMalloc((void**)&s,(k+1)*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&a_,k*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&b_,k*sizeof(double)));


    for(i=0;i<k+1;i++){
        CHECK_CUDA(cudaMalloc((void**) &Q_T[i],N*sizeof(double)));
        CHECK_CUDA(cudaMalloc((void**) &H[i],k*sizeof(double)));        
    } 



    cusparseHandle_t     sparse_handle = NULL;
    cusparseSpMatDescr_t A_double;
    cusparseSpMatDescr_t A_float;
    cusparseDnVecDescr_t xi_double;
    cusparseDnVecDescr_t Axi_double;
    cusparseCreate(&sparse_handle);


    CHECK_CUDA(cudaMemcpy(gpu_b_double, b, N*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_xi_double, xi, N*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_ri_double, xi, N*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_A_outerIndexPtr, A_outerIndexPtr, A_num_outerIndexPtr*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_A_innerIndexPtr, A_innerIndexPtr, A_num_innerIndexPtr*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_A_valuePtr_float, A_valuePtr_float, A_num_valuePtr*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_A_valuePtr_double, A_valuePtr_double, A_num_valuePtr*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_A_num_outerIndexPtr, &A_num_outerIndexPtr, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_A_num_innerIndexPtr, &A_num_innerIndexPtr, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_A_num_valuePtr, &A_num_valuePtr,  sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_N, &N, sizeof(int), cudaMemcpyHostToDevice));


    CHECK_CUSPARSE(cusparseCreateCsr(&A_double, N, N, A_num_valuePtr,
                                        gpu_A_outerIndexPtr, gpu_A_innerIndexPtr, gpu_A_valuePtr_double,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));


    CHECK_CUSPARSE(cusparseCreateDnVec(&xi_double,
                        N,
                        gpu_xi_double,
                        CUDA_R_64F));

    CHECK_CUSPARSE(cusparseCreateDnVec(&Axi_double,
                        N,
                        gpu_Axi_double,
                        CUDA_R_64F));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(sparse_handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &one_double,
                            A_double, 
                            xi_double,
                            &zero_double,
                            Axi_double,
                            CUDA_R_64F,
                            CUSPARSE_SPMV_CSR_ALG2,
                            &bufferSize));

    cudaMalloc(&externalBuffer, bufferSize);

    cusparseSpMV_preprocess(sparse_handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &one_double,
                            A_double, 
                            xi_double,
                            &zero_double,
                            Axi_double,
                            CUDA_R_64F,
                            CUSPARSE_SPMV_CSR_ALG2,
                        externalBuffer);
    //FP64 Residual
    CHECK_CUSPARSE(cusparseSpMV(sparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one_double,
        A_double,
        xi_double,
        &zero_double,
        Axi_double,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        externalBuffer));
    cudaMemcpy(gpu_ri_double, gpu_b_double, N*sizeof(double), cudaMemcpyDeviceToDevice);

    cublasDaxpy(handle, N,
            &neg_one_double,
            gpu_Axi_double, 1,
            gpu_ri_double, 1);


    cublasDnrm2(handle,N,gpu_ri_double,1,&prev_err);
    double actual_r0_norm=prev_err;


    //Bnorm
    CHECK_CUBLAS(cublasDnrm2(handle,N,gpu_b_double,1,&b_norm_double));

    b_norm=b_norm_double;
    int restart_ind;
    double sjp1;
    for(restart_ind=0;restart_ind<max_restarts;restart_ind++){
        CHECK_CUDA(cudaMemset(s, 0, (k+1)*sizeof(double)));


        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        if(prev_err>=tol*b_norm_double){
            #ifdef PRINT_RNORM
            cout<<restart_ind<<": "<<k<<" "<<prev_err/b_norm_double<<" "<<cr<<endl;
            #endif
            /*
            GMRES START
            */

            r0_norm=prev_err;
            
            put_vec64_div_scalar64_to_vec64<<<N,1>>>( Q_T[0],  gpu_ri_double, r0_norm, N);

                CHECK_CUDA(cudaDeviceSynchronize());
            beta = r0_norm;
            CHECK_CUDA(cudaMemcpy(s,&beta,sizeof(double),cudaMemcpyHostToDevice));
            k=30;

            for(j=0;j<k;j++){
                cusparseDnVecDescr_t Q_Tj_float;
                cusparseDnVecDescr_t Q_Tjp1_float;
                CHECK_CUSPARSE(cusparseCreateDnVec(&Q_Tj_float,
                                    N,
                                    Q_T[j],
                                    CUDA_R_64F));
                CHECK_CUSPARSE(cusparseCreateDnVec(&Q_Tjp1_float,
                                    N,
                                    Q_T[j+1],
                                    CUDA_R_64F));



                CHECK_CUSPARSE(cusparseSpMV(sparse_handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one_double,
                    A_double,
                    Q_Tj_float,
                    &zero_double,
                    Q_Tjp1_float,
                    CUDA_R_64F,
                    CUSPARSE_SPMV_CSR_ALG2,
                    externalBuffer));
                for(i=0;i<j+1;i++){
                    CHECK_CUBLAS(cublasDdot(handle, N, Q_T[i], 1,Q_T[j+1], 1,H[i]+j));
                    vec_sub_scalar_vec<<<grid_size,block_size>>>(Q_T[j+1],Q_T[i],H[i]+j,gpu_N);

                }
                CHECK_CUBLAS(cublasDnrm2(handle, N,Q_T[j+1], 1, H[j+1]+j));
                rescale_Q_Tjp1_if_needed<<<N,1>>>(H[j+1],tol,Q_T[j+1],N,j);
                CHECK_CUDA(cudaDeviceSynchronize());
                for(i=0;i<j;i++){
                    updateH1<<<N,1>>>(H[i],H[i+1],a_[i],b_[i],j);
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
                last_updates<<<N,1>>>(H[j],H[j+1],a_,b_,s,j,gpu_sjp1);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(&sjp1,gpu_sjp1,sizeof(double),cudaMemcpyDeviceToHost));
                if(sjp1<tol*b_norm_double){
                    for(i=0;i<k+1;i++){
                        CHECK_CUDA(cudaMemcpy(Q_T_local[i],Q_T[i],N*sizeof(double),cudaMemcpyDeviceToHost));
                        CHECK_CUDA(cudaMemcpy(H_local[i],H[i],k*sizeof(double),cudaMemcpyDeviceToHost));
                    } 
                    CHECK_CUDA(cudaMemcpy(s_local,s,(k+1)*sizeof(double),cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(xi,gpu_xi_double,N*sizeof(double),cudaMemcpyDeviceToHost));
                
                    CHECK_CUDA(cudaDeviceSynchronize());
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&tmp_gpu_time,start,stop);
gpu_time+=tmp_gpu_time;
                    for(i=j-1;i>=0;i--){
                        y[i]=s_local[i];
                        for(l=i+1;l<j;l++){
                            y[i]-=H_local[i][l]*y[l];
                        }
                        y[i]/=H_local[i][i];
                    }
                    for(i=0;i<N;i++){
                        for(l=0;l<j;l++){
                            xi[i]+=Q_T_local[l][i]*y[l];
                        }
                    }
                    restarts=restart_ind;
                    converged=true;
                    return total_iters+j+1;
                }
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            //least squares

            for(i=0;i<k+1;i++){
                CHECK_CUDA(cudaMemcpy(Q_T_local[i],Q_T[i],N*sizeof(double),cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(H_local[i],H[i],k*sizeof(double),cudaMemcpyDeviceToHost));
            } 
            CHECK_CUDA(cudaMemcpy(s_local,s,(k+1)*sizeof(double),cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(xi,gpu_xi_double,N*sizeof(double),cudaMemcpyDeviceToHost));
        
                CHECK_CUDA(cudaDeviceSynchronize());

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&tmp_gpu_time,start,stop);
gpu_time+=tmp_gpu_time;
            for(i=j-1;i>=0;i--){
                y[i]=s_local[i];
                for(l=i+1;l<j;l++){
                    y[i]-=H_local[i][l]*y[l];
                }
                y[i]/=H_local[i][i];
            }
            for(i=0;i<N;i++){
                for(l=0;l<j;l++){
                    xi[i]+=(double)Q_T_local[l][i]*(double)y[l];
                }
            }
            iters=j;
            res=fabs(s_local[iters]);
            if(isnan(res)){
                cout<<"Invalid! Exiting..."<<endl;
                exit(100);
            }
            if(res<=tol*b_norm){
                converged=true;
            }
            else{
                converged=false;
            }
cudaEventRecord(start);
            cudaMemcpy(gpu_xi_double, xi,N*sizeof(double),cudaMemcpyHostToDevice);



            CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));


            CHECK_CUSPARSE(cusparseCreateDnVec(&xi_double,
                                N,
                                gpu_xi_double,
                                CUDA_R_64F));

            CHECK_CUSPARSE(cusparseCreateDnVec(&Axi_double,
                                N,
                                gpu_Axi_double,
                                CUDA_R_64F));




            //FP64 Residual
            cusparseSpMV(sparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one_double,
                A_double,
                xi_double,
                &zero_double,
                Axi_double,
                CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                externalBuffer);
            cudaMemcpy(gpu_ri_double, gpu_b_double, N*sizeof(double), cudaMemcpyDeviceToDevice);
            cublasDaxpy(handle, N,
                    &neg_one_double,
                    gpu_Axi_double, 1,
                    gpu_ri_double, 1);
            cublasDnrm2(handle,N,gpu_ri_double,1,&cur_err);
            
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&tmp_gpu_time,start,stop);
gpu_time+=tmp_gpu_time;

            if(isnan(cur_err)){
                cout<<"Invalid! Exiting..."<<endl;
                exit(100);
            }
            cr=cur_err/prev_err;
            total_iters+=iters;
            if(cr>max_cr){
                cur_m = orig_k;
            }
            else if(cr<min_cr){
                cur_m = prev_m;
            }
            else{
                if(k-d>=d){
                    cur_m=k-d;
                }
                else{
                    cur_m=orig_k;
                }
            }
            prev_err=cur_err;
            prev_m=cur_m;
cudaEventRecord(start);
        }
        else{
            restarts=restart_ind;
            converged=true;
            return total_iters;
        }
    }

cudaEventRecord(start);
    cusparseDestroySpMat(A_double);
    cusparseDestroySpMat(A_float);
    cusparseDestroy(sparse_handle);

    cudaFree(gpu_y);
    cudaFree(gpu_z);
    cudaFree(gpu_i);
    cudaFree(gpu_j);
    cudaFree(gpu_b_double);
    cudaFree(gpu_xi_double);
    cudaFree(gpu_Axi_double);
    cudaFree(gpu_tmp_double);
    cudaFree(gpu_ri_double);
    cudaFree(gpu_A_outerIndexPtr);
    cudaFree(gpu_A_innerIndexPtr);
    cudaFree(gpu_A_valuePtr_float); 
    cudaFree(gpu_A_valuePtr_double); 
    cudaFree(gpu_A_num_outerIndexPtr);
    cudaFree(gpu_A_num_innerIndexPtr);
    cudaFree(gpu_A_num_valuePtr);
    cudaFree(gpu_nrm_double);
    cudaFree(gpu_N);
    cudaFree(externalBuffer);
    
    for(i=0;i<k+1;i++){
        cudaFree(Q_T[i]);
    } 

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&tmp_gpu_time,start,stop);
gpu_time+=tmp_gpu_time;
    
    restarts=restart_ind;
    converged=false;
    return total_iters;
}






int main(int argc, char *argv[]){
    srand(0);
    int i,j,k,l,m,n,N;
	string filename = argv[1];
	string struct_name = "Problem";
    matvar_t *matStruct;
    matvar_t *matVar;
    mat_sparse_t* a;
	mat_t *mat = Mat_Open(filename.c_str(),MAT_ACC_RDONLY);
    matStruct = Mat_VarRead(mat, struct_name.c_str());
    int num_fields = matStruct->nbytes/matStruct->data_size;
    for(i=0;i<num_fields;i++){
        matVar=Mat_VarGetStructFieldByIndex(matStruct, i, 0 );
        if(matVar->name[0]=='A' && matVar->name[1]=='\0'  ){
            break;
        }
    }
    a =(mat_sparse_t*) matVar->data;
    int A_num_row_ind = a->nir;
    int A_num_col_ptr = a->njc;
    int A_num_vals = a->ndata;
    int* A_row_ind = (int*) malloc(A_num_row_ind*sizeof(int));
    int* A_col_ptr = (int*) malloc(A_num_col_ptr*sizeof(int));
    double* A_vals = (double*) malloc(A_num_vals*sizeof(double));
    memcpy(A_row_ind,a->ir,A_num_row_ind*sizeof(int));
    memcpy(A_col_ptr,a->jc,A_num_col_ptr*sizeof(int));
    memcpy(A_vals,a->data,a->ndata*sizeof(double));

    Mat_Close(mat);
    Mat_VarFree(matStruct);

    N=A_num_col_ptr-1;
    SparseMatrix<double> A_csc(N,N);
    vector<Triplet<double>> tripletList;
    for (i = 0; i < N; i++) {
        for (j = A_col_ptr[i]; j < A_col_ptr[i+1]; j++) {
            tripletList.emplace_back(A_row_ind[j], i, A_vals[j]);
        }
    }
    A_csc.setFromTriplets(tripletList.begin(), tripletList.end());
    free(A_row_ind);
    free(A_col_ptr);
    free(A_vals);
    SparseMatrix<double, RowMajor> A = A_csc;
    A.makeCompressed();
    A.makeCompressed();
    int* A_outerIndexPtr= A.outerIndexPtr();
    int* A_innerIndexPtr= A.innerIndexPtr();
    double* A_valuePtr= A.valuePtr();
    int A_num_outerIndexPtr= A.outerSize()+1;
    int A_num_valuePtr= A.nonZeros();
    int A_num_innerIndexPtr= A_num_valuePtr;
    double* A_valuePtr_float=(double*) malloc(A_num_valuePtr*sizeof(double));
    for(i=0;i<A_num_valuePtr;i++){
        A_valuePtr_float[i]=A_valuePtr[i];
    }

    double* b = (double*)  malloc(N*sizeof(double));
    for(i=0;i<N;i++){b[i]=(((double)rand())/RAND_MAX);}
    double* x0 = (double*) calloc(N,sizeof(double));
    double* x_hat = (double*) calloc(N,sizeof(double));
    k=stoi(argv[2]); 
    double tol= stod(argv[3]);
    int max_restarts= stoi(argv[4]);
    int iters;
    int restarts;
    bool converged = false;
    double res;
    float gpu_time=0.0;
    cout<<"A is "<<N<<" x "<<N<<endl;
    warmup<<<256,1>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
auto start = high_resolution_clock::now();
    
    int total_iters = dynamic_restarted_gmres(
        A_outerIndexPtr,
        A_innerIndexPtr,
        A_valuePtr,
        A_valuePtr_float,
        A_num_outerIndexPtr,
        A_num_innerIndexPtr,
        A_num_valuePtr,
        b,
        x0,
        N,
        k,
        max_restarts,
        tol,
        x_hat, 
        iters,
        restarts,
        converged,
        res,
        gpu_time);
auto stop = high_resolution_clock::now();
    if(converged){
        cout<<"Converged within "<<total_iters<<" iterations"<<endl;
    }
    else{
        cout<<"Did not converge within "<<total_iters<<" iterations"<<endl;
    }
    cout<<endl;


    auto duration = duration_cast<microseconds>(stop - start);
    cout << "GMRES Time: "<< ((double)duration.count())*1e-6 << endl;
    cout << "GPU Time: "<<gpu_time*1e-3f<<endl;

    
}