import nksr
import open3d as o3d
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from pycg import vis

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str,
                        help="Input KITTI-like directory for data."
                        "Must contain a velodyne folder and poses.txt file")
    parser.add_argument("-v", "--voxel-size", type=float,
                        help="Final mesh voxel size", default=0.2)
    parser.add_argument("-c", "--chunk-size", type=float,
                        help="Chunk size, -1 indicates no chunking",
                        default=50.0)
    parser.add_argument("-o", "--output-file", type=str,
                        help="Output mesh file", required=False, default=None)

    args = parser.parse_args()
    voxel_size = args.voxel_size
    chunk_size = args.chunk_size
    output_filename = args.output_file

    input_dir_base = Path(args.input_dir)
    input_cloud_dir = input_dir_base / "velodyne"
    input_cloud_files = sorted(list(input_cloud_dir.glob("*.bin")))
    print(f"Found {len(input_cloud_files)} clouds to process.")

    traj_file = input_dir_base / "poses.txt"

    sensor_poses = []
    with open(traj_file) as f:
        lines = f.readlines()
    for line in lines:
        pose_vect = np.array([float(x) for x in line.split()])
        pose = pose_vect.reshape(3, 4)
        pose = np.vstack((pose, [0, 0, 0, 1]))
        sensor_poses.append(pose)

    print(f"Found {len(sensor_poses)} sensor poses.")

    # Prepare input data for NKSR
    # Uncomment the following two lines to subsample the input dataset.
    # Might needed if dataset is too big
    input_cloud_files = input_cloud_files[::3]
    sensor_poses = sensor_poses[::3]
    # Uncomment the following two lines to subsample the first 200 clouds
    # input_cloud_files = input_cloud_files[:200]
    # sensor_poses = sensor_poses[:200]

    cloud_np = None
    poses_np = None
    for i in tqdm(range(len(input_cloud_files))):
        cloud_partial_np = np.fromfile(
            input_cloud_files[i], "<f4").reshape(-1, 4)[..., :3]
        pose = sensor_poses[i]
        cloud_partial_np = cloud_partial_np @ pose[:3, :3].T + pose[:3, -1]
        # generate corresponding pose vect
        pose_partial = np.tile(
            sensor_poses[i][:3, -1], (cloud_partial_np.shape[0], 1))
        if cloud_np is None:
            cloud_np = cloud_partial_np.copy()
            poses_np = pose_partial.copy()
        else:
            cloud_np = np.vstack([cloud_np, cloud_partial_np])
            poses_np = np.vstack([poses_np, pose_partial])

    device = torch.device("cuda:0")
    cloud = torch.from_numpy(cloud_np).float().to(device)
    poses = torch.from_numpy(poses_np).float().to(device)
    if chunk_size > 0:
        scaling_factor = 0.1 / voxel_size
        print(f"Scaling factor: {scaling_factor}")
        scaled_origin = torch.mean(cloud, 0).cpu().numpy()
        cloud *= scaling_factor
        print(scaled_origin)

    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")

    field = reconstructor.reconstruct(cloud, sensor=poses,
                                      chunk_size=chunk_size,
                                      voxel_size=voxel_size,
                                      preprocess_fn=nksr.get_estimate_normal_preprocess_fn(
                                          64, 85.0))

    print(f"Extracting mesh with voxel_size={voxel_size}")
    mesh = field.extract_dual_mesh()
    mesh_o3d = vis.mesh(mesh.v, mesh.f)
    if chunk_size > 0:
        mesh_o3d.scale(1./scaling_factor, np.zeros((3,)))
        ...
    if output_filename is not None:
        print(f"Saving mesh to {output_filename}")
        vis.to_file(mesh_o3d, output_filename)
    vis.show_3d([mesh_o3d])

    exit(0)
