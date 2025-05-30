# 3D Assignments

**This project fulfills the assignment requirements of VDT.**

This repository provides scripts to render and process a 3D cow mesh using PyTorch3D and Kaolin. The tasks include mesh rendering, creating a turntable animation, applying camera transformations, converting to a point cloud, and voxelizing the mesh.

This repository provides scripts to render and process a 3D cow mesh using PyTorch3D and Kaolin. The tasks include mesh rendering, creating a turntable animation, applying camera transformations, converting to a point cloud, and voxelizing the mesh.
GitHub Repository: https://github.com/aquarter147/3D_assignments

## Dependencies

Install required packages via pip:

```bash
pip install -r requirements.txt
```


## Usage

1. **Render front‐view mesh**

   ```bash
   python render_cow.py
   ```

   * Produces `output/cow_rotation.jpg`.

2. **Create 360° mesh turntable**

   ```bash
   python render_cow_turntable.py
   ```

   * Produces `output/cow_turntable.gif`.

3. **Apply camera transformation**

   ```bash
   python render_cow_transformed.py
   ```

   * Prints `R_relative` and `T_relative`.
   * Produces `output/cow_transformed_view.png`.

4. **Mesh → Point Cloud**

   ```bash
   python mesh2pc.py
   ```

 * Produces `output/mesh2pc.gif`.

5. **Mesh → Voxel**

   ```bash
   python mesh2voxel.py
   ```

   * Produces `output/mesh2voxel_static.png`.
   * Creates `output/mesh2voxel_turntable.gif`.
