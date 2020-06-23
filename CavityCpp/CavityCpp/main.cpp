//
//  2D FDM Cavity Flow in Cpp
//  CavityCpp
//
//  Created by Aska on 2020/06/22.
//  Copyright © 2020 aska. All rights reserved.
//

#include <iostream>
#include "vtr_writer.hpp"
void InitialCondition(double* u,
                      double* v,
                      double* p,
                      double* s,
                      int nx,
                      int ny,
                      int c){
    for(int i = c; i < nx-c; i++)
        for(int j = c; j < ny-c; j++){
            int index = j * nx + i;
                u[index] = 0;
                v[index] = 0;
                p[index] = 0;
                s[index] = 0;
        }
}
void DirichletBoundary(double* u,
                       double* v,
                       int nx,
                       int ny,
                       int c){
    int i, j, index;
    double Nu, Nv;
    //East & West Halo
    for (int j = c; j < ny-c; j++) {
        i = 0;
        index = j * nx + i;
        //Nu = -u[index + 1];
        //Nv = -v[index + 1];
        Nu = 0;
        Nv = 0;
        
        u[index] = Nu;
        v[index] = Nv;
        
        i = nx - 1;
        index = j * nx + i;
        //Nu = -u[index - 1];
        //Nv = -v[index - 1];
        Nu = 0;
        Nv = 0;
        
        u[index] = Nu;
        v[index] = Nv;
    }
    //South & North Halo
    for (int i = c; i < nx-c; i++) {
        j = 0;
        index = j * nx + i;
        //Nu = -u[index + nx];
        //Nv = -v[index + nx];
        Nu = 0;
        Nv = 0;
        
        u[index] = Nu;
        v[index] = Nv;
        
        j = ny - 1;
        index = j * nx + i;
        //Nu = 2.0 -u[index - nx];
        //Nv =   -v[index - nx];
        Nu = 1;
        Nv = 0;
        u[index] = Nu;
        v[index] = Nv;
    }
}
void NeumannBoundary(double* p,
                     int nx,
                     int ny,
                     int c){
    int i, j, index;
    double Np;
    //East & West Halo
    for (int j = c; j < ny-c; j++) {
        i = 0;
        index = j * nx + i;
        Np = p[index + 1];
        p[index] = Np;
        
        i = nx - 1;
        index = j * nx + i;
        Np = p[index - 1];
        p[index] = Np;
    }
    //South & North Halo
    for (int i = c; i < nx-c; i++) {
        j = 0;
        index = j * nx + i;
        Np = p[index + nx];
        p[index] = Np;
        
        j = ny - 1;
        index = j * nx + i;
        //Np =  p[index - nx];
        Np = 0;
        p[index] = Np;
    }
    
}
void PoissonSource(double* s,
                   double* u,
                   double* v,
                   int nx,
                   int ny,
                   int c,
                   double dx,
                   double dy,
                   double dt,
                   double rho){
    double Ns;
    int index;
    for(int i = c; i < nx-c; i++)
        for(int j = c; j < ny-c; j++){
            index = j * nx + i;
            Ns = rho*( 1./dt *( (u[index+1]-u[index-1])/(2*dx) + (v[index+nx]-v[index-nx])/(2*dy) )
                      -((u[index+1]-u[index-1])/(2*dx)) * ((u[index+1]-u[index-1])/(2*dx))
                      -2 * ((u[index+nx]-u[index-nx])/(2*dy)) * ((v[index+1]-v[index-1])/(2*dx))
                      -((v[index+nx]-v[index-nx])/(2*dy)) * ((v[index+nx]-v[index-nx])/(2*dy)) );
            s[index] = Ns;
            if(index==nx*(ny-2)+20) std::cout<<u[index]<<std::endl;
        }
}
void PressurePossion(double* pn,
                     double* p,
                     double* s,
                     int nx,
                     int ny,
                     int c,
                     double dx,
                     double dy){
    double Np;
    for(int i = c; i < nx-c; i++)
        for(int j = c; j < ny-c; j++){
            int index = j * nx + i;
            Np =  ((p[index+1]+p[index-1])*(dy*dy)+(p[index+nx]+p[index-nx])*(dx*dx))/(2*(dx*dx+dy*dy))  -    s[index]*(dx*dx*dy*dy)/(2*(dx*dx+dy*dy));
            pn[index] = Np;
        }
}
void MomentumEq(double* un,
                double* vn,
                double* u,
                double* v,
                double* p,
                int nx,
                int ny,
                int c,
                double dx,
                double dy,
                double dt,
                double nu,
                double rho){
    double Nu, Nv;
    for(int i = c; i < nx-c; i++)
        for(int j = c; j < ny-c; j++){
            int index = j * nx + i;
            Nu = u[index] - dt*(u[index] * (u[index] - u[index - 1])/dx + v[index] * (u[index] - u[index - nx])/dy
                                +(p[index+1]-p[index-1])/(2*dx*rho)
                                -nu*( (u[index+1]-2*u[index]+u[index-1])/(dx*dx) + (u[index+nx]-2*u[index]+u[index-nx])/(dy*dy) ) );
            
            Nv = v[index] - dt*(u[index] * (v[index] - v[index - 1])/dx + v[index] * (v[index] - v[index - nx])/dy
                                +(p[index+nx]-p[index-nx])/(2*dy*rho)
                                -nu*( (v[index+1]-2*v[index]+v[index-1])/(dx*dx) + (v[index+nx]-2*v[index]+v[index-nx])/(dy*dy) ) );
            un[index] = Nu;
            vn[index] = Nv;
        }
}

void Projection(double* un,
                double* vn,
                double* u,
                double* v,
                double* p,
                int nx,
                int ny,
                int c,
                double dx,
                double dy,
                double dt,
                double rho){
    double Nu, Nv;
    for(int i = c; i < nx-c; i++)
        for(int j = c; j < ny-c; j++){
            int index = j * nx + i;
            Nu = u[index] - dt*(p[index+1]-p[index-1])/(2*dx*rho);
            Nv = v[index] - dt*(p[index+nx]-p[index-nx])/(2*dy*rho);
                                
            un[index] = Nu;
            vn[index] = Nu;
        }
}

void output_vtr(
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
    sprintf(filename, "Cavity");
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
    for(int j=jst; j<=jen; j++){ y.push_back( (j - jst+0.5)*dx ); }
    for(int i=ist; i<=ien; i++){ x.push_back( (i - ist+0.5)*dy ); }
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
int main(int argc, const char * argv[]) {
    //Load parameters
    const int nx = 41;
    const int ny = 41;
    const int N = nx*ny;
    const int nt = 500;
    const int nit = 100;
    const int c = 1;
    
    const double dx = 2. / (nx - 1);
    const double dy = 2. / (ny - 1);
    const double rho = 1.0;
    const double nu = 0.1;
    const double dt = 0.001;
    double u[N]= {0};
    double v[N]= {0};
    double p[N]= {0};
    double s[N]= {0};
    double un[N] = {0};
    double vn[N] = {0};
    double pn[N] = {0};
    
    std::cout<<"dx="<<dx<<dy<<std::endl;
    std::cout<<"rho="<<rho<<std::endl;
    std::cout<<"dt="<<dt<<std::endl;
    std::cout<<"nu="<<nu<<std::endl;
    
    InitialCondition(u, v, p, s, nx, ny, c);
    DirichletBoundary(u, v, nx, ny, c);
    NeumannBoundary(p, nx, ny, c);
    for (int i=0; i<nt; i++) {
        double simTime = dt* i;
        if(i%10==0)
        output_vtr(nx, ny, c, i/5, dx, dy, simTime, u, v, p);
        PoissonSource(s, u, v, nx, ny, c, dx, dy, dt, rho);
        //std::cout<<"s[middle]="<<s[nx*(ny-2)+20]<<std::endl;
        for (int j=0; j<nit; j++) {
            PressurePossion(pn, p, s, nx, ny, c, dx, dy);
            NeumannBoundary(pn, nx, ny, c);
            //SwapVar(pn, p, nx, ny);
            std::swap(pn,p);
            //std::cout<<"p[middle]="<<p[nx*ny/2]<<std::endl;
        }
        MomentumEq(un, vn, u, v, p, nx, ny, c, dx, dy, dt, nu, rho);
        DirichletBoundary(un, vn, nx, ny, c);
        std::swap(un,u);
        std::swap(vn,v);
        //std::cout<<"u[middle]="<<u[nx*ny/2]<<std::endl;
    }
    return 0;
}
