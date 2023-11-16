import pyrealsense2 as rs
pc = rs.pointcloud()
points = pc.calculate(depth_frame)
pc.map_to(color_frame)