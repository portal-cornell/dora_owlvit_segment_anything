import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

def publish_transforms():
    rospy.init_node('static_tf2_broadcaster')
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    transform1 = TransformStamped()
    transform1.header.stamp = rospy.Time.now()
    transform1.header.frame_id = "frame1"
    transform1.child_frame_id = "frame2"
    transform1.transform.translation.x = 1.0
    transform1.transform.translation.y = 2.0
    transform1.transform.translation.z = 3.0
    transform1.transform.rotation.x = 0.0
    transform1.transform.rotation.y = 0.0
    transform1.transform.rotation.z = 0.0
    transform1.transform.rotation.w = 1.0

    transform2 = TransformStamped()
    transform2.header.stamp = rospy.Time.now()
    transform2.header.frame_id = "frame3"
    transform2.child_frame_id = "frame2"
    transform2.transform.translation.x = 4.0
    transform2.transform.translation.y = 5.0
    transform2.transform.translation.z = 6.0
    transform2.transform.rotation.x = 0.0
    transform2.transform.rotation.y = 0.0
    transform2.transform.rotation.z = 0.0
    transform2.transform.rotation.w = 1.0

    broadcaster.sendTransform([transform1, transform2])

    rate = rospy.Rate(1)
    rospy.spin()

if __name__ == '__main__':
    publish_transforms()