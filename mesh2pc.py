import os
import torch
import imageio

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    PointsRenderer, PointsRasterizer, PointsRasterizationSettings,
    AlphaCompositor, FoVPerspectiveCameras, look_at_view_transform,
    PointLights
)
from pytorch3d.structures import Pointclouds

def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def mesh2pc(
    mesh_path="data/cow.obj",
    num_samples=10000,
    image_size=512,
    dist=2.0,
    elev=0.0,
    azim_steps=36,
    background_color=(1.0, 1.0, 1.0),
    point_color=(0.4, 0.4, 0.8),
):
    device = get_device()
    mesh = load_objs_as_meshes([mesh_path], device=device)
    pts = sample_points_from_meshes(mesh, num_samples)  # (1, N, 3)
    
    colors = torch.tensor(point_color, device=device).view(1,1,3).expand_as(pts)
    pc = Pointclouds(points=pts, features=colors)

    rast_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.005,            
        points_per_pixel=10      
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=rast_settings),
        compositor=AlphaCompositor(background_color=background_color)
    )

    images = []
    for azim in torch.linspace(0, 360, azim_steps):
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = PointLights(location=[[0,0,-3]], device=device)

        rend = renderer(pc, cameras=cameras, lights=lights)
        img = (rend[0, ..., :3].cpu().numpy() * 255).astype("uint8")
        images.append(img)
    return images

def save_gif(images, out_path="mesh2pc.gif", fps=10):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, images, fps=fps)
    print(f"Saved point-cloud GIF to {out_path}")

if __name__ == "__main__":
    imgs = mesh2pc(
        mesh_path="data/cow.obj",
        num_samples=20000,      
        image_size=512,
        dist=2.5,
        elev=30.0,              
        azim_steps=60
    )
    save_gif(imgs, "output/mesh2pc.gif")
