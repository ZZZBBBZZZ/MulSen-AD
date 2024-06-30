import open3d as o3d
import numpy as np
import os
import glob

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point  
      

def stl2pcd(file_path):
    # root_folder = os.getcwd()
    
    # root_folder = '/ssd2/m3lab/usrs/ljq/tools/3DAD/'
    mesh_stl = o3d.geometry.TriangleMesh()
 
    # load ply
    mesh_stl = o3d.io.read_triangle_mesh(file_path)
    mesh_stl.compute_vertex_normals()
 
    # V_mesh 为ply网格的顶点坐标序列，shape=(n,3)，这里n为此网格的顶点总数，其实就是浮点型的x,y,z三个浮点值组成的三维坐标
    V_mesh = np.asarray(mesh_stl.vertices)
    # F_mesh 为ply网格的面片序列，shape=(m,3)，这里m为此网格的三角面片总数，其实就是对顶点序号（下标）的一种组合，三个顶点组成一个三角形
    F_mesh = np.asarray(mesh_stl.triangles)
    N = V_mesh.shape[0]
    if N >= 16384:
        pointcloud = farthest_point_sample(V_mesh,16384)
    else:
        pointcloud = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_stl,number_of_points=16384)
        pointcloud = np.array(pointcloud.points)
    # o3d.visualization.draw_geometries([mesh_ply], window_name="ply", mesh_show_wireframe=True)
 
    # ply -> stl
    # mesh_stl = o3d.geometry.TriangleMesh()
    # mesh_stl.vertices = o3d.utility.Vector3dVector(V_mesh)
    # mesh_stl.triangles = o3d.utility.Vector3iVector(F_mesh)
    # mesh_stl.compute_vertex_normals()
 
    # stl/ply -> pcd
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pointcloud)
    print("pcd info:",pcd)
    # o3d.visualization.draw_geometries([pcd],window_name="pcd")
 
    # save pcd
    # print(file_path.replace('ply','pcd'))
    o3d.io.write_point_cloud(file_path.replace('stl','pcd'),pcd)

if __name__ == '__main__':
    # ply2pcd('/ssd2/m3lab/usrs/ljq/tools/3DAD/1502_final.ply')
    # ply_list = glob.glob('/ssd2/m3lab/data/3DAD/3dad_demo_ply/*/*.ply')+glob.glob('/ssd2/m3lab/data/3DAD/3dad_demo_ply/*/*/*.ply')
    # ply_list = sorted(glob.glob('/ssd2/m3lab/data/3DAD/3dad_demo_ply/test/normal/*.ply'))[-10:]
    # ply_list = sorted(glob.glob('/ssd2/m3lab/data/3DAD/3dad_demo_ply/test/normal/*.ply'))
    # ply_list = sorted(glob.glob('/ssd2/m3lab/data/3DAD/3dad_demo_more_ply/*/*.ply') + glob.glob('/ssd2/m3lab/data/3DAD/3dad_demo_more_ply/*/*/*.ply)'))
    mulsen = [
        "capsule",
        "cube",
        "cotton",
            ]
    for cls in mulsen:
        stl_list = sorted(glob.glob('/home/liwq/datasets/mulsen/{}/pc/stl/train/*.stl'.format(cls)))
        for stl_file in stl_list:
            stl2pcd(stl_file)