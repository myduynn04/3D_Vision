import os
os.makedirs("output", exist_ok=True)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights,
    look_at_view_transform, TexturesVertex
)
from render_mesh import get_mesh_renderer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mesh = load_objs_as_meshes(["data/cow.obj"], device=device)

verts_rgb = torch.tensor([[[0.7, 0.7, 1.0]]], device=device).expand_as(mesh.verts_padded())
mesh.textures = TexturesVertex(verts_rgb)
renderer = get_mesh_renderer(image_size=512, device=device)
lights = PointLights(location=[[0, 0, -3]], device=device)


R0 = torch.eye(3, device=device).unsqueeze(0)           # (1,3,3)
T0 = torch.tensor([[0.0, 0.0, 3.0]], device=device)    # (1,3)


R_new, T_new = look_at_view_transform(dist=3.0, elev=90.0, azim=0.0, device=device)

R_rel = R_new @ R0.transpose(1,2)
T_rel = T_new - (R_rel @ T0.unsqueeze(-1)).squeeze(-1)

print("R_relative:\n", R_rel)
print("T_relative:\n", T_rel)


R_t = R_rel @ R0
T_t = (R_rel @ T0.unsqueeze(-1)).squeeze(-1) + T_rel
cameras_t = FoVPerspectiveCameras(R=R_t, T=T_t, fov=60, device=device)
img = renderer(mesh, cameras=cameras_t, lights=lights)[0, ..., :3].cpu().numpy()

plt.imsave("output/cow_transformed_view.png", img)
print("Saved output/cow_transformed_view.png")
