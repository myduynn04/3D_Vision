import os
os.makedirs("output", exist_ok=True)
os.environ.pop("MPLBACKEND", None)

import torch
import matplotlib
matplotlib.use("Agg")
import imageio

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights,
    look_at_view_transform, TexturesVertex
)
from render_mesh import get_mesh_renderer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mesh = load_objs_as_meshes(["data/cow.obj"], device=device)

base_color = torch.tensor([[[0.7, 0.7, 1.0]]], device=device)
verts_rgb = base_color.expand_as(mesh.verts_padded())
mesh.textures = TexturesVertex(verts_rgb)

renderer = get_mesh_renderer(image_size=512, device=device)

images = []
for azim in torch.linspace(0, 360, steps=36):
    R, T = look_at_view_transform(dist=2.0, elev=0.0, azim=azim, device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, fov=60.0, device=device)
    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    img = (rend[0, ..., :3].cpu().numpy() * 255).astype('uint8')
    images.append(img)

output_path = "output/cow_turntable.gif"
imageio.mimsave(output_path, images, fps=15)
print(f"GIF saved as {output_path}")
