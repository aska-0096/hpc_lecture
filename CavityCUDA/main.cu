#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vtr_writer.hpp"
#include <iostream>
// Cuda Error Guardian
#define cudaCheckErrors(msg) \
    do { \
        cudaDeviceSynchronize(); \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            return 1; \
        } \
    } while (0)

void output_vtr(
//output function
    const int NX,
    const int NY,
    const int Halo,
          int outputStep,
    const double dx,
    const double dy,       
          double simTime, 
          double* H_U, 
          double* H_V, 
          double* H_P)
{
    // file name & dir path setting
    char output_dir[128], filename[128], dir_path[128];
    sprintf(output_dir, ".");
    sprintf(filename, "CavityCUDA");
    sprintf(dir_path, "%s/%s", output_dir, filename);

    bool is_Halo_output = true;
    
    int bound = (is_Halo_output)? 0 : Halo;
    // make buffers for cell centered data
    std::vector<double> buff_p;
    std::vector<double> buff_u, buff_v, buff_w;
    std::vector<double> x,y,z;
    const int ist = bound, ien = NX-bound;
    const int jst = bound, jen = NY-bound;
    // set coordinate
    for(int j=jst; j<=jen; j++){ y.push_back( (j - jst)*dx ); }
    for(int i=ist; i<=ien; i++){ x.push_back( (i - ist)*dy ); }
    z.push_back( 0.0 );
    // write cell data
    for(int j = jst; j < jen; j++)
        for(int i = ist; i < ien; i++)
        {
            int index=j*NX+i;
            buff_p.push_back(H_P[index] );
            //using cell averaged velocity 
            buff_u.push_back(H_U[index]);
            buff_v.push_back(H_V[index]);
            buff_w.push_back(0);            
        }
    flow::vtr_writer vtr;
    vtr.init(dir_path,filename, 
        NX-2*bound+1, NY-2*bound+1, 1,
         0, NX-2*bound, 0, NY-2*bound, 0, 1,
          true);
    vtr.set_coordinate(&x.front(),&y.front(),&z.front());
    vtr.push_cell_array("Velocity", &buff_u.front(), &buff_v.front(), &buff_w.front());
    vtr.push_cell_array("Pressure", &buff_p.front());
    vtr.set_current_step(outputStep);
    vtr.write(simTime);
}
//MemUsage
void MemUsage()
{
    size_t freeMem, totalMem;
    float GigaBytes = 1024.0f*1024.0f*1024.0f;
    cudaMemGetInfo( &freeMem, &totalMem );
    std::cout << "Memory Usage: " << (totalMem - freeMem)/GigaBytes << " / " << totalMem/GigaBytes << " GB" << std::endl;
}
//kernel 1
//Momentum Equation
__global__
void convection(
    const int NX,
    const int NY,
	const int Halo,
    const double DX,
    const double DY,
    const double rho,
    const double nu,
    const double DT,
    double * U,
    double * Un,
    double * V,
    double * Vn,
    double * P
    )
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
    int index=j*NX+i;
    double Nu, Nv;
    if(i>=Halo && i<NX-Halo && j>=Halo && j<NY-Halo)
    {
	    Nu = U[index] - DT*(U[index] * (U[index] - U[index - 1])/DX + V[index] * (U[index] - U[index - NX])/DY
                        +(P[index+1]-P[index-1])/(2*DX*rho)
                        -nu*( (U[index+1]-2*U[index]+U[index-1])/(DX*DX) + (U[index+NX]-2*U[index]+U[index-NX])/(DY*DY) ) );
        Nv = V[index] - DT*(U[index] * (V[index] - V[index - 1])/DX + V[index] * (V[index] - V[index - NX])/DY
                        +(P[index+NX]-P[index-NX])/(2*DX*rho)
                        -nu*( (V[index+1]-2*V[index]+V[index-1])/(DX*DX) + (V[index+NX]-2*V[index]+V[index-NX])/(DY*DY) ) );
        Un[index] = Nu;
        Vn[index] = Nv;
	}
}
//kernel 2
//calculate the source term of pressure
__global__
void Scal(
	const int NX,
    const int NY,
	const int Halo,
    const double DX,
    const double DY,
    const double DT,
    const double rho,
    double * S,
    double * U,
    double * V
	)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
    int index=j*NX+i;
    double Ns;
    if(i>=Halo && i<NX-Halo && j>=Halo && j<NY-Halo){
        Ns = rho*( 1./DT *( (U[index+1]-U[index-1])/(2*DX) + (V[index+NX]-V[index-NX])/(2*DY) )
                      -((U[index+1]-U[index-1])/(2*DX)) * ((U[index+1]-U[index-1])/(2*DX))
                      -2 * ((U[index+NX]-U[index-NX])/(2*DY)) * ((V[index+1]-V[index-1])/(2*DX))
                      -((V[index+NX]-V[index-NX])/(2*DY)) * ((V[index+NX]-V[index-NX])/(2*DY)) );
        S[index] = Ns;
    }
}
//kernel 3
//function to update the pressure
__global__
void Piter(
	const int NX,
    const int NY,
	const int Halo,
    const double DX,
    const double DY,
    double * S,
    double * P,
    double * Pn
	)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
    int index=j*NX+i;
    double Np;
    if(i>=Halo && i<NX-Halo && j>=Halo && j<NY-Halo){
        Np =  ((P[index+1]+P[index-1])*(DY*DY)+(P[index+NX]+P[index-NX])*(DX*DX))/(2*(DX*DX+DY*DY))  
               -S[index]*(DX*DX*DY*DY)/(2*(DX*DX+DY*DY));
        Pn[index] = Np;
    }
}
//Kernel 7
//write boundary condition in u, v respectively
//Non-slip wall for velocity
__global__
void EWBound(
    const int NX,
    const int NY,
    const int Halo,
          double* U,
          double* V
           )
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int index=j*NX+i;

    double Nu, Nv;
    //east
    Nu = -U[index + 1];
    Nv = -V[index + 1];
    U[index] = 0;
    V[index] = 0;
    //west
    index = j*NX + NX-1-i;
    Nu = -U[index - 1];
    Nv = -V[index - 1];
    U[index] = 0;
    V[index] = 0;
}
__global__
void SNBound(
	const int NX,
    const int NY,
	const int Halo,
	      double* U,
          double* V	
           )
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
    int index=j*NX+i;

    double Nu, Nv;
    //south
    Nu = -U[index + NX];
    Nv = -V[index + NX];
    U[index] = 0;
    V[index] = 0;
    //north
    index = (NY-j-1)*NX + i;
    Nu = 2.0 -U[index - NX];
    Nv = -V[index - NX];
    U[index] = 1;
    V[index] = 0;
}
//Kernel 8
//Neumann boundary condition for pressure
__global__
void EWPBound(
    const int NX,
    const int NY,
    const int Halo,
          double* P
    )
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int index=j*NX+i;

    double Np;
    //east
    Np = P[index + 1];
    P[index] = Np;
    //west
    index = j*NX + NX-1-i;
    Np = P[index - 1];
    P[index] = Np;
}
__global__
void SNPBound(
    const int NX,
    const int NY,
    const int Halo,
          double* P
    )
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int index=j*NX+i;

    double Np;
    //east
    Np = P[index + NX];
    P[index] = Np;
    //west
    index = (NY-j-1)*NX + i;
    Np = P[index - NX];
    P[index] = Np;

}
//Kernel 9
//initialization function
__global__
void initialization(
	const int NX,
	const int NY,
	const int Halo,
	double* U,
	double* V,
	double* P
	)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int index=j*NX+i;
    if (i<NX && j<NY )
    {
        //North Halo
        if(j>NY-Halo-1 && i>=Halo && i<NX-Halo)
        {
            U[index]=1;
            V[index]=0;
            P[index]=0;
        }
        else
        {
            U[index]=0;
            V[index]=0;
            P[index]=0; 
        }
    }
}
int main(int argc, char *argv[]){

    
	const int NX=41,
	          NY=41,
	          Halo=1;

	      int nt=500,
	          simTime=0,
              nit=50;

	const double LX=2,
	             LY=2,
	             DX=LX/NX,
	             DY=LY/NY,
	             rho=1,       //density
                 nu=0.1,
                 dt=0.001;        
    //Grid division
    dim3 Block(NX,1,1);
	dim3 Grid( (NX+Block.x-1)/Block.x,
	           (NY+Block.y-1)/Block.y,1);
    //EW division
    dim3 EWBlock(Halo,NY,1);
    dim3 EWGrid( 1,1,1);
    //SN division
    dim3 SNBlock(NX,Halo,1);
    dim3 SNGrid(1, 1, 1);

	size_t size = NX * NY *sizeof(double);
	//CPU pointer
	double* H_U,
	      * H_V,
	      * H_P;
	//Allocate CPU memory
    H_U  = (double*)malloc(size);
    H_V  = (double*)malloc(size);
    H_P  = (double*)malloc(size);
    //GPU pointer
    double* U,      //velocity in X direction
          * Un,    //intermediate X velocity
          * V,      //velocity in Y direction
          * Vn,    //intermediate Y velocity
          * P,      //pressure
          * Pn,    //intermediate Pressure
          * S;       //laplacian of P
    //Allocate GPU memory;
    cudaMalloc((void**) &U , size);
    cudaMalloc((void**) &Un , size);
    cudaMalloc((void**) &V , size);
    cudaMalloc((void**) &Vn , size);
    cudaMalloc((void**) &P , size);
    cudaMalloc((void**) &Pn , size);
    cudaMalloc((void**) &S , size);
    //initialization
	initialization<<<Grid,Block>>>(NX,NY,Halo,U,V,P);
    cudaCheckErrors("Kernel launch failure");
    //check Memory allocation
    MemUsage();
	for(int i=0; i<nt; i++)
	{
        if(i%5==0){
            //copy data from device to host
            cudaMemcpy(H_U , U , size, cudaMemcpyDeviceToHost);
            cudaCheckErrors("CUDA memcpy failure");
            cudaMemcpy(H_V , V , size, cudaMemcpyDeviceToHost);
            cudaCheckErrors("CUDA memcpy failure");
            cudaMemcpy(H_P , P , size, cudaMemcpyDeviceToHost);
            cudaCheckErrors("CUDA memcpy failure");
            //output velocity and pressure data to vtr file
            simTime=i*dt;
            output_vtr(NX, NY, Halo, i/5, DX, DY, simTime, H_U, H_V, H_P);
        } 
        //cacluate the source term for virtual pressure
        Scal<<<Grid,Block>>>( NX, NY, Halo, DX, DY, dt, rho, S, Un, Vn);
        cudaCheckErrors("Kernel launch failure");    
        // intern step to get virtual pressure
        for(int j=0; j<nit; j++)
        {
            Piter<<<Grid,Block>>>( NX, NY, Halo, DX, DY, S, P, Pn);
            cudaCheckErrors("Kernel launch failure");           
            EWPBound<<<EWGrid,EWBlock>>>(NX, NY, Halo, Pn);
            cudaCheckErrors("Kernel launch failure");
            SNPBound<<<SNGrid,SNBlock>>>(NX, NY, Halo, Pn);
            cudaCheckErrors("Kernel launch failure");
            std::swap(Pn,P);
        }
        //RK3 to calculate convection term
		convection<<<Grid,Block>>>(NX, NY, Halo, DX, DY, rho, nu, dt, U, Un, V, Vn, P);
		cudaCheckErrors("Kernel launch failure");
        EWBound<<<EWGrid,EWBlock>>>(NX, NY, Halo, Un, Vn);
        cudaCheckErrors("Kernel launch failure");
        SNBound<<<SNGrid,SNBlock>>>(NX, NY, Halo, Un, Vn);
        cudaCheckErrors("Kernel launch failure");
        std::swap(Un,U);
        std::swap(Vn,V);
	}
	//free GPU memory
	cudaFree(U);
	cudaFree(Un);
	cudaFree(V);
	cudaFree(Vn);
	cudaFree(P);
	cudaFree(Pn);
	cudaFree(S);
    //free CPU memory
	free(H_U);
	free(H_V);
    free(H_P);
    return 0;
}

