import rospy

import tf.transformations as tr
# import tf2_ros.
import tf2_ros
import geometry_msgs.msg
import numpy as np
# from scipy.spatial.transform import Rotation as R
CHILD_FRAME_ID = "mocap"
PARENT_FRAME_ID = "panda_link0"

cam2tag_transform = {
	"translation": [0, 0, 3.0],
	"rotation": [[-0.6932634  , 0.69451749 ,-0.19243523],
 [-0.4848414 , -0.64701774, -0.58846993],
 [-0.53321167 ,-0.3146641 , 0.78528455]]}



# [0.03970234, -0.27958421, 1.35536968]
# [[-0.6932634   0.69451749 -0.19243523]
#  [-0.4848414  -0.64701774 -0.58846993]
#  [-0.53321167 -0.3146641   0.78528455]]

# [[-0.83343159, 0.54600808, -0.08524648],
# 					[-0.26791058, -0.53412818, -0.80182979],
# 					[-0.48333809, -0.64543184,  0.59144064]]
tag2world_transform = {
	"translation": [0, -1.147, -0.965],
	"rotation": [0, 0, 0]}
### z is 96.5
### y is 114.7
### x is 0 
# pose_R = [[-0.83343159  0.54600808 -0.08524648]
#  [-0.26791058 -0.53412818 -0.80182979]
#  [-0.48333809 -0.64543184  0.59144064]]
# pose_t = [[ 0.03970234]
#  [-0.27958421]
#  [ 1.35536968]]

# flipped image
# [ 0.87445213, -0.45247517,  0.17492769],
#  [ 0.14987644,  0.59493978,  0.78967317],
#  [-0.46137894, -0.66431385,  0.58806172]

cam2tag_transform = {
	"translation": [ 0.29127413, -0.23409795, 1.72393682],
	"rotation": [[-0.86756514 , 0.45619762 ,-0.19802643],
 [-0.14420667 ,-0.61183811 ,-0.77772653],
 [-0.47595711, -0.64617169 , 0.59659616]]}

pose_R = []

matrix_mult = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

print(np.dot(matrix_mult, cam2tag_transform['rotation'], ))

# cam2tag_transform = {
# 	"translation": [0, 0, 3.5],
# 	"rotation": [[-0.86642793 , 0.46014262 ,-0.19383345],
#  [0.14438694, 0.60252376, 0.78493154],
#  [0.47796972 ,0.65209959 , -0.588482  ]]}

# [-0.6932634, -0.7203755, -0.0210924],
#  [-0.4848414,  0.4445403,  0.7532017],
#  [-0.5332116,  0.5323936, -0.6574514  ]

# [-0.86642793 , 0.46014262 ,-0.19383345],
#  [-0.14438694, -0.60252376, -0.78493154],
#  [-0.47796972 ,-0.65209959 , 0.588482  ]


if __name__ == '__main__':
	rospy.init_node('mocap_to_world_transform')
	broadcaster = tf2_ros.StaticTransformBroadcaster()
	cam2tag = geometry_msgs.msg.TransformStamped()

	cam2tag.header.stamp = rospy.Time.now()
	cam2tag.header.frame_id = "cam_1_color_optical_frame"
	cam2tag.child_frame_id = "mocap"

	cam2tag.transform.translation.x = cam2tag_transform['translation'][0]
	cam2tag.transform.translation.y = cam2tag_transform['translation'][1]
	cam2tag.transform.translation.z = cam2tag_transform['translation'][2]

	# eul = [0, 0, 0 ]
	# r = R.from_euler("xyz", eul, degrees=True)

	# # [ -21.83601257   32.22271232 -145.03255815]
	# # r = R.from_matrix((cam2tag_transform['rotation']))
	# eul = r.as_euler('xyz', degrees=True)
	# print(eul)
	# quat = r.as_quat()
	
	x = np.array(cam2tag_transform['rotation'])
	print(x)
	# x = np.linalg.inv(x)
	# print(x)
	# import pdb; pdb.set_trace()
	R = tr.random_rotation_matrix()
	R[:3, :3] = x
	quat = tr.quaternion_from_matrix(R)

	print(quat)
	print(sum([q**2 for q in quat]))
	# 180 rotation about x-axis
	# quat = tr.quaternion_multiply(tr.quaternion_about_axis(3.14159, (0,0,1)), quat)
	# quat = [ 0.3363192, -0.883958, -0.2440063, -0.214403 ]
	# quat = [ 0.3590827, -0.8392741, -0.3771614, -0.1562942 ]
	rotation_matrix = tr.quaternion_matrix(quat)
	# # Create a rotation matrix for the rotation about the x-axis
	# x_rotation_matrix = tr.rotation_matrix(np.pi, (1, 0, 0))
	# print(x_rotation_matrix)
	# # Apply the rotation and convert back to a quaternion
	x_rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	new_rotation_matrix = rotation_matrix@x_rot
	quat = tr.quaternion_from_matrix(new_rotation_matrix)
	print(quat)
	cam2tag.transform.rotation.x = quat[0]
	cam2tag.transform.rotation.y = quat[1]
	cam2tag.transform.rotation.z = quat[2]
	cam2tag.transform.rotation.w = quat[3]

	# tag2world = geometry_msgs.msg.TransformStamped()
     
	# tag2world.header.stamp = rospy.Time.now()
	# tag2world.header.frame_id = "tag"
	# tag2world.child_frame_id = "map"

	# tag2world.transform.translation.x = tag2world_transform['translation'][0]
	# tag2world.transform.translation.y = tag2world_transform['translation'][1]
	# tag2world.transform.translation.z = tag2world_transform['translation'][2]

	# tag2world.transform.rotation.x = 0
	# tag2world.transform.rotation.y = 0
	# tag2world.transform.rotation.z = 0
	# tag2world.transform.rotation.w = 1

	broadcaster.sendTransform(cam2tag)
	# broadcaster.sendTransform(tag2world)
	rate = rospy.Rate(1)
	rospy.spin()
