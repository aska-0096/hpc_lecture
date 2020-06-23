#ifndef FLOW_VTR_WRITER_H
#define FLOW_VTR_WRITER_H

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>

// for mkdir
#ifdef _MSC_VER
# include <direct.h>
#else // _MSC_VER
# include <sys/stat.h>
#endif // _MSC_VER

namespace flow {

namespace detail {
/**
 *  typename to VTK typename literal
 */
template<typename T>
struct vtr_type {
    static const char* name() { return "Undefined"; }
};
#define FLOW_VTK_TYPE_DEF(TYPE, NAME) \
template<> \
struct vtr_type<TYPE> { \
    static const char* name() { return NAME; } \
}
FLOW_VTK_TYPE_DEF(char,   "Int8");
FLOW_VTK_TYPE_DEF(short,  "Int16");
FLOW_VTK_TYPE_DEF(int,    "Int32");
FLOW_VTK_TYPE_DEF(long,   "Int64");
FLOW_VTK_TYPE_DEF(float,  "Float32");
FLOW_VTK_TYPE_DEF(double, "Float64");

FLOW_VTK_TYPE_DEF(const char,   "Int8");
FLOW_VTK_TYPE_DEF(const short,  "Int16");
FLOW_VTK_TYPE_DEF(const int,    "Int32");
FLOW_VTK_TYPE_DEF(const long,   "Int64");
FLOW_VTK_TYPE_DEF(const float,  "Float32");
FLOW_VTK_TYPE_DEF(const double, "Float64");

FLOW_VTK_TYPE_DEF(unsigned char,   "UInt8");
FLOW_VTK_TYPE_DEF(unsigned short,  "UInt16");
FLOW_VTK_TYPE_DEF(unsigned int,    "UInt32");
FLOW_VTK_TYPE_DEF(unsigned long,   "UInt64");

FLOW_VTK_TYPE_DEF(const unsigned char,   "UInt8");
FLOW_VTK_TYPE_DEF(const unsigned short,  "UInt16");
FLOW_VTK_TYPE_DEF(const unsigned int,    "UInt32");
FLOW_VTK_TYPE_DEF(const unsigned long,   "UInt64");
#undef FLOW_VTK_TYPE_DEF

} // namespace detail


/**
 *  VTK RectilinearGrid Writer for arrays
 */
class vtr_writer
{
    typedef const char byte_t;

    // coordinate type    
    typedef float coord_t;
    
    // unsigned 32bit integer type
    typedef int UINT32_t;

    class holder_t
    {
    public:
        struct Data {
            byte_t*   data;
            byte_t*   data2;
            byte_t*   data3;
            size_t  size_of;
            bool is_vector;
            std::string VTK_typename;
        };
        
        holder_t() {}
        ~holder_t() {}

        /**
         *  add pointer
         */
        template<class T> void push(const std::string& name, T* ptr);
        template<class T> void push(const std::string& name, T* u, T* v, T* w);
        /**
         *  reset pointer
         */
        template<class T> void reset(const std::string& name, T* ptr);
        template<class T> void reset(const std::string& name, T* u, T* v, T* w);

        inline Data& data(int i) { return data_[i]; }
        inline const Data& data(int i) const { return data_[i]; }
        inline size_t size() const { return data_.size(); }

        inline const std::string& name(int i) const { return names_[i]; }
        inline size_t        size_of  (int i) const { return data_[i].size_of; }
        inline bool          is_vector(int i) const { return data_[i].is_vector; }
        inline const char*   type_name(int i) const { return data_[i].VTK_typename.c_str(); }
        inline const byte_t* ptr      (int i) { return data_[i].data; }
        inline const byte_t* u        (int i) { return data_[i].data; }
        inline const byte_t* v        (int i) { return data_[i].data2; }
        inline const byte_t* w        (int i) { return data_[i].data3; }


    private:
        std::vector<Data>        data_;
        std::vector<std::string> names_;

        /// @return index of name
        inline int findIdx(const std::string& name) const;
    };

public:

    vtr_writer()
      : point_data_(NULL)
      , cell_data_(NULL)
      , serial_(0)
      , continue_(false)
      , d2f_(false)
    {
        for(int i=0; i<3; i++) x_[i] = NULL;
    }

    ~vtr_writer()
    {
        for(int i=0; i<3; i++) delete[] x_[i];
        delete point_data_;
        delete cell_data_;
    }

    void init
    (
        const std::string& directory,
        const std::string& name_core,
        int Nx, int Ny, int Nz,
        int ist, int ied,
        int jst, int jed,
        int kst, int ked,
        bool double2float = false
    ) {
        set_output_directory_root(directory);
        set_name_core(name_core);

        d2f_ = double2float;

        Nx_[0] = Nx; Nx_[1] = Ny; Nx_[2] = Nz;
        nx_[0] = ied - ist;
        nx_[1] = jed - jst;
        nx_[2] = ked - kst;
        ist_[0] = ist; ist_[1] = jst; ist_[2] = kst;
        ied_[0] = ied; ied_[1] = jed; ied_[2] = ked;

        for(int i=0; i<3; i++) {
            rank_[i] = 0;
            ncpu_[i] = 1;
            gnx_[i] = nx_[i];
            gNx_[i] = Nx_[i];
        }
        for(int ii=0; ii<3; ii++) {
            x_[ii] = new coord_t[Nx_[ii]];
            for(int i=0; i<Nx_[ii]; i++) x_[ii][i] = i;
        }
    }

    // setter
    void enable_continue() { continue_ = true; }
    void set_output_directory_root(const std::string& directory) { directory_ = directory; }
    void set_name_core(const std::string& name_core) { name_core_ = name_core; }
    void set_current_step(int step) { serial_ = step; }
    void set_mpi_info(const int (&rank)[3], const int (&ncpu)[3]) 
    {
        for(int i=0; i<3; i++) {
            rank_[i] = rank[i];
            ncpu_[i] = ncpu[i];
            gnx_[i] = ncpu_[i]*nx_[i];
            gNx_[i] = gnx_[i] + ist_[i] + Nx_[i] - ied_[i];
        }
    }

    int step() const { return serial_; }
    
    // set coordinate
    template<typename T> void set_coordinate(const T* x, const T* y);
    template<typename T> void set_coordinate(const T* x, const T* y, const T* z);
    template<typename T> void create_coordinate(const T (&x0)[3], const T (&dx)[3]);

    // add array to output
    template<typename T> void push_point_array (const std::string& name, T* ptr);
    template<typename T> void push_point_array (const std::string& name, T* u, T* v, T* w);
    template<typename T> void reset_point_array(const std::string& name, T* ptr);
    template<typename T> void reset_point_array(const std::string& name, T* u, T* v, T* w);
    template<typename T> void push_cell_array  (const std::string& name, T* ptr);
    template<typename T> void push_cell_array  (const std::string& name, T* u, T* v, T* w);
    template<typename T> void reset_cell_array (const std::string& name, T* ptr);
    template<typename T> void reset_cell_array (const std::string& name, T* u, T* v, T* w);

    /**
     *  make a .vtr file 
     */
    void write
    (
        double time,
        int downsize = 1
    ) {
        if(downsize <= 0) downsize = 1;

        std::ostringstream oss;
        oss << directory_;
        if(downsize >= 2) oss << "-ds" << downsize;
        const std::string& dir = oss.str();

        mkdir(dir.c_str());

        std::FILE* fp = std::fopen(vtr_name(dir, serial_, downsize).c_str(), "w");
        size_t vtr_offset = 0;

        // downsizing
        int dNx[3], ist[3], exL[3], exR[3];
        for(int i=0; i<3; i++) {
            downsize_offset(&dNx[i], &ist[i], &exL[i], &exR[i], Nx_[i], gnx_[i], downsize, rank_[i], ncpu_[i]);
        }

        std::fprintf(fp, "<?xml version=\"1.0\"?>\n");
        std::fprintf(fp, "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

        std::fprintf(fp, "\t<RectilinearGrid WholeExtent=\"%d %d %d %d %d %d\">\n", exL[0], exR[0], exL[1], exR[1], exL[2], exR[2]);

        std::fprintf(fp, "\t\t<Piece Extent=\"%d %d %d %d %d %d\">\n",              exL[0], exR[0], exL[1], exR[1], exL[2], exR[2]);
        if(point_data_) write_data_arrays(fp, "PointData", *point_data_, dNx, &vtr_offset);
        if(cell_data_)  write_data_arrays(fp, "CellData",  *cell_data_,  dNx, &vtr_offset);
        write_coordinate(fp, dNx, &vtr_offset);
        std::fprintf(fp, "\t\t</Piece>\n");

        std::fprintf(fp, "\t</RectilinearGrid>\n");

        std::fprintf(fp, "<AppendedData encoding=\"raw\">\n_");
        if(point_data_) write_data_arrays_binary(fp, dNx, ist, downsize, *point_data_);
        if(cell_data_)  write_data_arrays_binary(fp, dNx, ist, downsize, *cell_data_);
        write_coodinate_binary(fp, dNx, ist, downsize);
        std::fprintf(fp, "\n</AppendedData>\n");

        std::fprintf(fp, "</VTKFile>\n");
        std::fclose(fp);

        make_header(serial_, time, downsize);
        if(downsize == 1) serial_++;
    }

    /**
     *  make a .pvtr file & add .pvd file
     */
    void make_header
    (
        int    file_num,
        double time,
        int    downsize = 1
    ) {
        const int rank = rank_[0] + ncpu_[0]*(rank_[1] + ncpu_[1]*rank_[2]);
        if(rank == 0) {
            if(downsize <= 0) downsize = 1;
            std::ostringstream oss;
            oss << directory_;
            if(downsize >= 2) oss << "-ds" << downsize;
            const std::string& dir = oss.str();
            mkdir(dir.c_str());

            // make pvtr
            std::FILE* fp = std::fopen(pvtr_name(dir, file_num, downsize).c_str(), "w");

            std::fprintf(fp, "<?xml version=\"1.0\"?>\n");

            std::fprintf(fp, "<VTKFile type=\"PRectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

            // downsize 
            const int gdNx[3] = {
                downsize_cast(gNx_[0], downsize),
                downsize_cast(gNx_[1], downsize),
                downsize_cast(gNx_[2], downsize)
            };
            std::fprintf(fp, "\t<PRectilinearGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n", gdNx[0]-1, gdNx[1]-1, gdNx[2]-1);

            if(point_data_) write_parallel_data_arrays(fp, "PPointData", *point_data_);
            if(cell_data_)  write_parallel_data_arrays(fp, "PCellData",  *cell_data_);
            write_parallel_coordinate(fp);

            const int ncpu_x = ncpu_[0];
            const int ncpu_y = ncpu_[1];
            const int ncpu_z = ncpu_[2];

            for(int rank_z=0; rank_z<ncpu_z; rank_z++)
            for(int rank_y=0; rank_y<ncpu_y; rank_y++)
            for(int rank_x=0; rank_x<ncpu_x; rank_x++) {
                const int rank = rank_x + ncpu_x*(rank_y + ncpu_y*rank_z);
                const int ranks[3] = { rank_x, rank_y, rank_z };
                
                // downsizing
                int dNx[3], ist[3], exL[3], exR[3];
                for(int ii=0; ii<3; ii++) {
                    downsize_offset(&dNx[ii], &ist[ii], &exL[ii], &exR[ii], Nx_[ii], gnx_[ii], downsize, ranks[ii], ncpu_[ii]);
                }

                std::fprintf(fp, "\t\t<Piece Extent=\"%d %d %d %d %d %d\" Source=\"%s\" />\n",
                            exL[0], exR[0], exL[1], exR[1], exL[2], exR[2],
                            filename(".", name_core_, "vtr", file_num, downsize, rank).c_str()
                        );
            }

            std::fprintf(fp, "\t</PRectilinearGrid>\n");
            std::fprintf(fp, "</VTKFile>\n");
            std::fclose(fp);

            // make pvd
            std::ofstream ofs;
            std::ostringstream os;
            const std::string pname = pvd_name(dir, downsize).c_str();
            std::ofstream ofs_check(pname.c_str(), std::ios_base::in);

            if(!continue_ || ofs_check.fail()) {
                ofs.open(pname.c_str());
                ofs << "<?xml version=\"1.0\"?>\n"
                       "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                       "\t<Collection>\n";
                continue_ = true;
            } else { 
                ofs_check.close();
                // append!
                ofs.open(pname.c_str(), std::ios_base::in);
                ofs.seekp(-25, std::ios_base::end);
            }
            ofs << "\t\t<DataSet timestep=\"" << time <<"\" group=\"\" part=\"0\" file=\"" << pvtr_name_here(file_num, downsize) <<"\"/>\n"
                   "\t</Collection>\n"
                   "</VTKFile>";
        }
    }

private:
    std::string directory_; // output directory
    std::string name_core_;  // name core

    // geometry
    int Nx_[3]; // number of cell center
    int nx_[3]; // number of cell center
    int ist_[3];
    int ied_[3];
    int gNx_[3];
    int gnx_[3];

    // MPI
    int rank_[3];
    int ncpu_[3];
    
    holder_t* point_data_;
    holder_t* cell_data_;
    coord_t* x_[3]; // coordinate
    
    int serial_;

    bool continue_;
    bool d2f_; // convert double to float

    /**
     *  for downsizing
     */
    static inline int downsize_cast(int Nx, int downsize)
    {
        return (Nx + downsize - 1) / downsize;
    }
    static inline void downsize_offset
    (
        int* dNx,
        int* offset,
        int* extent_L,
        int* extent_R,
        int Nx, int gnx,
        int downsize, int rank, int ncpu
    ) {
        const int nx    = gnx / ncpu;
        const int gNx_L  = rank*nx;
        const int gdNx_L = downsize_cast(gNx_L, downsize);
        const int ist = gdNx_L * downsize;
        *offset = ist - gNx_L;
        *dNx = downsize_cast(Nx-*offset, downsize);
        *extent_L = gdNx_L;
        *extent_R = gdNx_L + *dNx - 1;
    }


    // filename utillity ----------
    
    /**
     *  file name creater
     */
    std::string filename
    (
        const std::string& directory,
        const std::string& name_core,
        const std::string& extension,
        int                number,
        int                downsize,
        int                rank
    ) {
        const int digit = 5; // maniau desho
        std::ostringstream ss;
                          ss << directory;         // dir
                          ss << '/' << name_core;  // dir-ds2/core
        if(downsize >= 2) ss << "-ds" << downsize; // dir-ds2/core-ds2
        if(rank >= 0)     ss << '-' << rank;       // dir-ds2/core-ds2-0
                          ss << '-' << std::setw(digit) << std::setfill('0') << number;
                          ss << '.' << extension;  // dir-ds2/core-ds2-0-00000.vtr
        return ss.str();
    }

    /**
     *  .vtr file name
     */
    std::string vtr_name(const std::string& dir, int step, int downsize)
    {
        const int rank = rank_[0] + ncpu_[0]*(rank_[1] + ncpu_[1]*rank_[2]);
        return filename(dir, name_core_, "vtr", step, downsize, rank);
    }

    /**
     *  .pvtr file name
     */
    std::string pvtr_name(const std::string& dir, int step, int downsize) 
    {
        return filename(dir, name_core_, "pvtr", step, downsize, -1);
    }

    /**
     *  .pvd file name
     */
    std::string pvd_name(const std::string& dir, int downsize) 
    {
        std::ostringstream oss;
                          oss << dir << '/' << name_core_;
        if(downsize >= 2) oss << "-ds" << downsize;
                          oss << ".pvd";
        return oss.str();
    }

    /**
     *  for .pvd
     */
    std::string pvtr_name_here(int step, int downsize) 
    {
        return filename(".", name_core_, "pvtr", step, downsize, -1);
    }

    // array name writer

    /**
     *  write array informations for .vtr
     */
    void write_data_arrays
    (
        std::FILE* fp,
        const char* datatype, 
        holder_t& holder,
        const int (&Nx)[3],
        size_t* offset
    ) {
        std::fprintf(fp, "\t\t\t<%s>\n", datatype);
        for(size_t i=0; i<holder.size(); i++) {
            const bool is_double = std::strcmp(holder.type_name(i),"Float64") == 0;
            const char* type_name = (d2f_ && is_double)? "Float32"     : holder.type_name(i);
            const size_t size_of =  (d2f_ && is_double)? sizeof(float) : holder.size_of(i);

            if(!holder.is_vector(i)) {
                std::fprintf(fp, "\t\t\t\t<DataArray type=\"%s\" Name=\"%s\" format=\"appended\" offset=\"%zu\" />\n",
                         type_name, holder.name(i).c_str(), *offset);
                *offset += sizeof(UINT32_t) + size_of*Nx[0]*Nx[1]*Nx[2];
            } else {
                std::fprintf(fp, "\t\t\t\t<DataArray type=\"%s\" NumberOfComponents=\"3\" Name=\"%s\" format=\"appended\" offset=\"%zu\" />\n",
                         type_name, holder.name(i).c_str(), *offset);
                *offset += sizeof(UINT32_t) + size_of*Nx[0]*Nx[1]*Nx[2]*3;
            }

        }
        std::fprintf(fp, "\t\t\t</%s>\n", datatype);
    }
    /**
     *  write array informations for .pvtr
     */
    void write_parallel_data_arrays
    (
        std::FILE* fp,
        const char* datatype, 
        holder_t& holder
    ) {
        std::fprintf(fp, "\t\t<%s>\n", datatype);
        for(size_t i=0; i<holder.size(); i++) {
            const bool is_double = std::strcmp(holder.type_name(i),"Float64") == 0;
            const char* type_name = (d2f_ && is_double)? "Float32" : holder.type_name(i);

            if(!holder.is_vector(i)) {
                std::fprintf(fp, "\t\t\t<PDataArray type=\"%s\" Name=\"%s\" format=\"appended\" />\n",
                         type_name, holder.name(i).c_str());
            } else {
                std::fprintf(fp, "\t\t\t<PDataArray type=\"%s\" NumberOfComponents=\"3\" Name=\"%s\" format=\"appended\" />\n",
                         type_name, holder.name(i).c_str());
            }
        }
        std::fprintf(fp, "\t\t</%s>\n", datatype);
    }

    /**
     *  write coordinate informations for .vtr
     */
    void write_coordinate
    (
        //File& f,
        std::FILE* fp,
        const int (&Nx)[3],
        size_t* offset
    ) {
        std::fprintf(fp, "\t\t\t<Coordinates>\n");

        std::fprintf(fp, "\t\t\t\t<DataArray type=\"Float32\" Name=\"X\" format=\"appended\" offset=\"%zu\" />\n", *offset);
        *offset += sizeof(UINT32_t) + sizeof(coord_t)*Nx[0];
        std::fprintf(fp, "\t\t\t\t<DataArray type=\"Float32\" Name=\"Y\" format=\"appended\" offset=\"%zu\" />\n", *offset);
        *offset += sizeof(UINT32_t) + sizeof(coord_t)*Nx[1];
        std::fprintf(fp, "\t\t\t\t<DataArray type=\"Float32\" Name=\"Z\" format=\"appended\" offset=\"%zu\" />\n", *offset);

        std::fprintf(fp, "\t\t\t</Coordinates>\n");
    }
    /**
     *  write coordinate informations for .pvtr
     */
    void write_parallel_coordinate
    (
        //File& f
        std::FILE* fp
    ) {
        std::fprintf(fp, "\t\t<PCoordinates>\n");
        std::fprintf(fp, "\t\t\t<PDataArray type=\"Float32\" Name=\"X\" format=\"appended\" />\n");
        std::fprintf(fp, "\t\t\t<PDataArray type=\"Float32\" Name=\"Y\" format=\"appended\" />\n");
        std::fprintf(fp, "\t\t\t<PDataArray type=\"Float32\" Name=\"Z\" format=\"appended\" />\n");
        std::fprintf(fp, "\t\t</PCoordinates>\n");
    }

    /**
     *  write array data to file
     */
    void write_data_arrays_binary
    (
        std::FILE* fp,
        const int (&Nx)[3],
        const int (&ist)[3],
        int downsize,
        holder_t& holder
    ) {
        const size_t NN = static_cast<size_t>(Nx[0])*static_cast<size_t>(Nx[1])*static_cast<size_t>(Nx[2]);

        for(size_t i=0; i<holder.size(); i++) {
            const bool is_double = std::strcmp(holder.type_name(i),"Float64") == 0;

            if(!holder.is_vector(i)) {
                UINT32_t size = holder.size_of(i)*NN;
                std::fwrite(&size, sizeof(UINT32_t), 1, fp);
                if(d2f_ && is_double) fwrite_scalar_float(fp, holder.ptr(i), holder.size_of(i), ist, downsize);
                else                  fwrite_scalar      (fp, holder.ptr(i), holder.size_of(i), ist, downsize);
            } else {
                UINT32_t size = holder.size_of(i)*NN*3;
                std::fwrite(&size, sizeof(UINT32_t), 1, fp);
                if(d2f_ && is_double) fwrite_vector_float(fp, holder.u(i), holder.v(i), holder.w(i), holder.size_of(i), ist, downsize);
                else                  fwrite_vector      (fp, holder.u(i), holder.v(i), holder.w(i), holder.size_of(i), ist, downsize);
            }
        }
    }

    /**
     *  write coordinates to file
     */
    void write_coodinate_binary
    (
        std::FILE* fp,
        const int (&Nx)[3],
        const int (&ist)[3],
        int downsize
    ) {
        UINT32_t size = 0;
        for(int i=0; i<3; i++) {
            size = sizeof(coord_t)*(Nx[i]);
            std::fwrite(&size, sizeof(UINT32_t), 1, fp);
            if(downsize == 1) {
                std::fwrite(x_[i], size, 1, fp);
            } else {
                for(int j=ist[i]; j<Nx_[i]; j+=downsize) {
                    std::fwrite(x_[i]+j, sizeof(coord_t), 1, fp);
                }
            }
        }
    }

    // binary writer ---------------------------

    void fwrite_scalar
    (
        std::FILE* fp,
        const byte_t* src,
        size_t sizeof_elem,
        const int (&ist)[3],
        int downsize
    ) {
        if(downsize == 1) {
            std::fwrite(src, sizeof_elem*Nx_[0]*Nx_[1]*Nx_[2], 1, fp);
        } else {
            for(int k=ist[2]; k<Nx_[2]; k+=downsize)
            for(int j=ist[1]; j<Nx_[1]; j+=downsize)
            for(int i=ist[0]; i<Nx_[0]; i+=downsize) {
                const int id = i + Nx_[0]*(j + Nx_[1]*k);
                std::fwrite(src + id*sizeof_elem, sizeof_elem, 1, fp);
            }
        }
    }

    void fwrite_scalar_float
    (
        std::FILE* fp,
        const byte_t* src,
        size_t sizeof_elem,
        const int (&ist)[3],
        int downsize
    ) {
        for(int k=ist[2]; k<Nx_[2]; k+=downsize)
        for(int j=ist[1]; j<Nx_[1]; j+=downsize)
        for(int i=ist[0]; i<Nx_[0]; i+=downsize) {
            const int id = i + Nx_[0]*(j + Nx_[1]*k);
            const float ff = static_cast<float>(*reinterpret_cast<const double*>(src + id*sizeof_elem));
            std::fwrite(&ff, sizeof(ff), 1, fp);
        }
    }
    void fwrite_vector
    (
        std::FILE* fp,
        const byte_t* u,
        const byte_t* v,
        const byte_t* w,
        size_t sizeof_elem,
        const int (&ist)[3],
        int downsize
    ) {
        for(int k=ist[2]; k<Nx_[2]; k+=downsize)
        for(int j=ist[1]; j<Nx_[1]; j+=downsize)
        for(int i=ist[0]; i<Nx_[0]; i+=downsize) {
            const int id = i + Nx_[0]*(j + Nx_[1]*k);
            std::fwrite(u + id*sizeof_elem, sizeof_elem, 1, fp);
            std::fwrite(v + id*sizeof_elem, sizeof_elem, 1, fp);
            std::fwrite(w + id*sizeof_elem, sizeof_elem, 1, fp);
        }
    }

    void fwrite_vector_float
    (
        std::FILE* fp,
        const byte_t* u,
        const byte_t* v,
        const byte_t* w,
        size_t sizeof_elem,
        const int (&ist)[3],
        int downsize
    ) {
        for(int k=ist[2]; k<Nx_[2]; k+=downsize)
        for(int j=ist[1]; j<Nx_[1]; j+=downsize)
        for(int i=ist[0]; i<Nx_[0]; i+=downsize) {
            const int id = i + Nx_[0]*(j + Nx_[1]*k);
            const float uu[3] = {
                static_cast<float>(*reinterpret_cast<const double*>(u + id*sizeof_elem)),
                static_cast<float>(*reinterpret_cast<const double*>(v + id*sizeof_elem)),
                static_cast<float>(*reinterpret_cast<const double*>(w + id*sizeof_elem))
            };
            std::fwrite(uu, sizeof(uu), 1, fp);
        }
    }

    /**
     *  make directory
     */
    static int mkdir(const char* dirpath)
    {
#ifdef _MSC_VER
        return ::_mkdir(dirpath);
#else // _MSC_VER
        std::istringstream iss(dirpath); 
        std::string tmp; 
        std::vector<std::string> dirs;
        while(std::getline(iss, tmp, '/')){
            dirs.push_back(tmp);
        }
        if(dirs.size() == 0) return 0;

        int ret;
        std::string path = dirs[0];
        for(size_t i=1; i<dirs.size(); i++) {
            //std::cout << path << std::endl;
            ret = ::mkdir(path.c_str(), S_IEXEC|S_IWRITE|S_IREAD);
            path += '/' + dirs[i];
        }
        ret = ::mkdir(path.c_str(), S_IEXEC|S_IWRITE|S_IREAD);
        return ret;
#endif // _MSC_VER
    }
};

template<typename T>
void vtr_writer::set_coordinate(const T* x, const T* y)
{
    for(int i=0; i<Nx_[0]; i++) x_[0][i] = static_cast<coord_t>(x[i]);
    for(int j=0; j<Nx_[1]; j++) x_[1][j] = static_cast<coord_t>(y[j]);
    x_[2][0] = 0.0;
}

template<typename T>
void vtr_writer::set_coordinate(const T* x, const T* y, const T* z)
{
    for(int i=0; i<Nx_[0]; i++) x_[0][i] = static_cast<coord_t>(x[i]);
    for(int j=0; j<Nx_[1]; j++) x_[1][j] = static_cast<coord_t>(y[j]);
    for(int k=0; k<Nx_[2]; k++) x_[2][k] = static_cast<coord_t>(z[k]);
}

template<typename T>
void vtr_writer::create_coordinate(const T (&x0)[3], const T (&dx)[3])
{
    for(int ii=0; ii<3; ii++) {
        for(int i=0; i<Nx_[ii]; i++) x_[ii][i] = static_cast<coord_t>(x0[ii] + (i - ist_[ii])*dx[ii]);
    }
}

template<typename T>
void vtr_writer::push_point_array(const std::string& name, T* ptr)
{
    if(!point_data_) point_data_ = new holder_t();
    point_data_->push(name, ptr);
}

template<typename T>
void vtr_writer::push_point_array(const std::string& name, T* u, T* v, T* w)
{
    if(!point_data_) point_data_ = new holder_t();
    point_data_->push(name, u, v, w);
}

template<typename T>
void vtr_writer::reset_point_array(const std::string& name, T* ptr)
{
    point_data_->reset(name, ptr);
}

template<typename T>
void vtr_writer::reset_point_array(const std::string& name, T* u, T* v, T* w)
{
    point_data_->reset(name, u, v, w);
}

template<typename T>
void vtr_writer::push_cell_array(const std::string& name, T* ptr)
{
    if(!cell_data_) cell_data_ = new holder_t();
    cell_data_->push(name, ptr);
}

template<typename T>
void vtr_writer::push_cell_array(const std::string& name, T* u, T* v, T* w)
{
    if(!cell_data_) cell_data_ = new holder_t();
    cell_data_->push(name, u, v, w);
}

template<typename T>
void vtr_writer::reset_cell_array(const std::string& name, T* ptr)
{
    cell_data_->reset(name, ptr);
}

template<typename T>
void vtr_writer::reset_cell_array(const std::string& name, T* u, T* v, T* w)
{
    cell_data_->reset(name, u, v, w);
}


// holder_t ------------------------------------------

/**
 *  add pointer
 */
template<class T>
void vtr_writer::holder_t::push(const std::string& name, T* ptr)
{
    Data data = { reinterpret_cast<byte_t*>(ptr), NULL, NULL, sizeof(T), false, detail::vtr_type<T>::name() };
    data_.push_back(data);
    names_.push_back(name);
}

/**
 *  add pointer (vector)
 */
template<class T>
void vtr_writer::holder_t::push(const std::string& name, T* u, T* v, T* w)
{
    Data data = { reinterpret_cast<byte_t*>(u), 
                  reinterpret_cast<byte_t*>(v), 
                  reinterpret_cast<byte_t*>(w), 
                  sizeof(T), true, detail::vtr_type<T>::name() };
    data_.push_back(data);
    names_.push_back(name);
}

/**
 *  returns index of name
 */
inline int vtr_writer::holder_t::findIdx(const std::string& name) const
{
    typedef std::vector<std::string>::const_iterator iterator;
    iterator begin = names_.begin();
    iterator end   = names_.end();
    iterator it = std::find(begin, end, name);
    if(it == end) {
        std::ostringstream ss;
        ss << "[flow::holder_t] error: Array name \"" << name << "\" is not Pushed";
        throw std::runtime_error(ss.str());
    }
    return it - begin;
}

/**
 *  reset pointer
 */
template<class T>
inline void vtr_writer::holder_t::reset(const std::string& name, T* ptr)
{
    Data& data = data_[findIdx(name)];
    if(data.is_vector) {
        std::ostringstream ss;
        ss << "[flow::vtr::vtr_writer::holder_t] '" << name << "' is vector!";
        throw std::runtime_error(ss.str());
    }
    data.data = reinterpret_cast<byte_t*>(ptr);
}
template<class T>
inline void vtr_writer::holder_t::reset(const std::string& name, T* u, T* v, T* w)
{
    Data& data = data_[findIdx(name)];
    if(!data.is_vector) {
        std::ostringstream ss;
        ss << "[flow::holder_t] '" << name << "' is vector!";
        throw std::runtime_error(ss.str());
    }
    data.data  = reinterpret_cast<byte_t*>(u);
    data.data2 = reinterpret_cast<byte_t*>(v);
    data.data3 = reinterpret_cast<byte_t*>(w);
}

}// namespace flow

#endif // FLOW_VTR_WRITER_H
