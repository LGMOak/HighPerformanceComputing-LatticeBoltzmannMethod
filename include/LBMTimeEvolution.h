#pragma once

#include "LBMConfig.h"
#include "LBMGrid.h"
#include <hdf5.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

namespace LBM {

class AnimationWriter {
private:
    hid_t file_id_;
    hid_t velocity_group_;
    const Grid* grid_;

public:
    AnimationWriter(const std::string& filename, const Grid& grid) : grid_(&grid) {
        // Create HDF5 file, overwriting if it exists
        file_id_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id_ < 0) {
            throw std::runtime_error("Failed to create HDF5 file: " + filename);
        }

        // Create a group to store velocity data
        velocity_group_ = H5Gcreate2(file_id_, "/velocity", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (velocity_group_ < 0) {
            H5Fclose(file_id_);
            throw std::runtime_error("Failed to create HDF5 group '/velocity'");
        }

        write_metadata();
    }

    ~AnimationWriter() {
        if (velocity_group_ >= 0) H5Gclose(velocity_group_);
        if (file_id_ >= 0) H5Fclose(file_id_);
    }

    void write_frame(int timestep) {
        const int nx = grid_->nx();
        const int ny = grid_->ny();

        // Prepare a flat vector for the velocity magnitude data
        std::vector<double> vel_mag_data(nx * ny);

        // Copy data from grid and calculate velocity magnitude
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                // IMPORTANT: Use row-major order (y * nx + x) for Python/NumPy compatibility
                const int idx = y * nx + x;
                const double ux_val = grid_->ux(x, y);
                const double uy_val = grid_->uy(x, y);
                vel_mag_data[idx] = std::sqrt(ux_val * ux_val + uy_val * uy_val);
            }
        }

        // Write just the velocity magnitude for this timestep to keep things simple
        std::string dset_name = "vel_mag_" + std::to_string(timestep);
        write_dataset_2d(velocity_group_, dset_name, vel_mag_data, ny, nx);
    }

private:
    void write_metadata() {
        // Write grid dimensions as a simple 1D array [nx, ny]
        hsize_t dims[1] = {2};
        hid_t space_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(file_id_, "/grid_dims", H5T_NATIVE_INT, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        int grid_dims[2] = {grid_->nx(), grid_->ny()};
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, grid_dims);

        H5Dclose(dset_id);
        H5Sclose(space_id);
    }

    // Simplified 2D dataset writer without extra options like compression
    void write_dataset_2d(hid_t group_id, const std::string& name, const std::vector<double>& data, int rows, int cols) {
        // HDF5 and NumPy use row-major order, so dimensions are {rows, cols}
        hsize_t dims[2] = {static_cast<hsize_t>(rows), static_cast<hsize_t>(cols)};
        hid_t space_id = H5Screate_simple(2, dims, NULL);

        hid_t dset_id = H5Dcreate2(group_id, name.c_str(), H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

        H5Dclose(dset_id);
        H5Sclose(space_id);
    }
};

}
