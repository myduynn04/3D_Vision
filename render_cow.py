import os
os.makedirs("output", exist_ok=True)
os.environ.pop("MPLBACKEND", None)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras, PointLights, TexturesVertex
from render_mesh import get_mesh_renderer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mesh = load_objs_as_meshes(["data/cow.obj"], device=device)
verts_rgb = torch.tensor([[[0.7,0.7,1.0]]], device=device)
verts_rgb = verts_rgb.expand_as(mesh.verts_padded())
mesh.textures = TexturesVertex(verts_rgb)


R = torch.eye(3, device=device).unsqueeze(0)
T = torch.tensor([[0,0,3]], device=device)
cameras = FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
lights   = PointLights(location=[[0,0,-3]], device=device)


renderer = get_mesh_renderer(image_size=512, device=device)


image = renderer(mesh, cameras=cameras, lights=lights)[0, ..., :3].cpu().numpy()
plt.imsave("output/cow_rotation.jpg", image)
print("Saved output/cow_rotation.jpg")
