import rospy
from std_msgs.msg import Float32

class HapticNode:
    def __init__(self):
        rospy.init_node('haptic_node', anonymous=True)

        # Subscribe to contact or force sensor topic from UR10e
        rospy.Subscriber('/ur10e/contact_force', Float32, self.force_callback)

        # Publisher for haptic feedback intensity
        self.pub = rospy.Publisher('/haptic/feedback', Float32, queue_size=10)

        rospy.loginfo("Haptic node started")
        rospy.spin()

    def force_callback(self, msg):
        # Simple mapping from force to haptic intensity
        intensity = msg.data * 0.8  # scale down to safe haptic range

        # Clamp intensity between 0 and 1
        if intensity > 1.0:
            intensity = 1.0
        elif intensity < 0.0:
            intensity = 0.0

        # Publish placeholder haptic feedback
        self.pub.publish(Float32(data=intensity))

if __name__ == '__main__':
    try:
        HapticNode()
    except rospy.ROSInterruptException:
        pass
