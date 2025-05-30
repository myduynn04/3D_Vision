import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

from pytorch3d.io import load_objs_as_meshes
import kaolin.ops.conversions as conversions


os.makedirs("output", exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mesh = load_objs_as_meshes(["data/cow.obj"], device=device)
verts = mesh.verts_padded()[0]   # (V, 3)
faces = mesh.faces_padded()[0]   # (F, 3)

resolution = 64
voxel_grid = conversions.trianglemeshes_to_voxelgrids(
    verts.unsqueeze(0), faces, resolution
)[0].cpu().numpy().astype(bool)


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(voxel_grid, facecolors='blue', edgecolor='k')
ax.view_init(elev=20, azim=30)
ax.set_axis_off()
plt.tight_layout()
static_path = "output/mesh2voxel_static.png"
plt.savefig(static_path, bbox_inches='tight', dpi=100)
plt.close(fig)
print(f"Saved static voxel image to {static_path}")


images = []
angles = np.linspace(0, 360, 36, endpoint=False)
for angle in angles:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, facecolors='blue', edgecolor='k')
    ax.view_init(elev=20, azim=float(angle))
    ax.set_axis_off()
    plt.tight_layout()
    temp_file = f"output/temp_voxel_{int(angle)}.png"
    plt.savefig(temp_file, bbox_inches='tight', dpi=100)
    plt.close(fig)
    images.append(imageio.imread(temp_file))
    os.remove(temp_file)

gif_path = "output/mesh2voxel_turntable.gif"
imageio.mimsave(gif_path, images, fps=10)
print(f"Saved voxel turntable GIF to {gif_path}")
