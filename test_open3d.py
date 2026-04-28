import os
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "true"

import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh.compute_vertex_normals()
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)

o3d.visualization.draw([mesh, frame], raw_mode=True)