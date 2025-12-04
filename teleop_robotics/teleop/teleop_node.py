import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy  # placeholder for Leap Motion input
from std_msgs.msg import Float32
from teleop.motion_mapper import map_hand_to_robot
from teleop.impedance_control import compute_velocity
from teleop.utils import log_command

class TeleopNode:
    def __init__(self):
        rospy.init_node('teleop_node', anonymous=True)

        # Publisher to robot velocity commands
        self.cmd_pub = rospy.Publisher('/ur10e/cmd_vel', Twist, queue_size=10)

        # Placeholder haptic feedback publisher
        self.haptic_pub = rospy.Publisher('/haptic/feedback', Float32, queue_size=10)

        # Subscribe to Leap Motion input
        rospy.Subscriber('/leap_motion/hand', Joy, self.hand_callback)

        # Store latest hand input
        self.hand_data = None

        # Load parameters (from params.yaml or defaults)
        self.linear_scale = rospy.get_param('~linear_scale', 0.5)
        self.stiffness = rospy.get_param('~stiffness', 50.0)
        self.damping = rospy.get_param('~damping', 20.0)

        rospy.loginfo("Teleop Node started")
        self.run()

    def hand_callback(self, msg):
        # Just store the latest message
        self.hand_data = msg

    def run(self):
        rate = rospy.Rate(20)  # 20 Hz
        while not rospy.is_shutdown():
            if self.hand_data:
                # Map hand data to desired robot displacement (2 DOF, forward/back)
                desired_x = map_hand_to_robot(self.hand_data, self.linear_scale)

                # Compute velocity using simple impedance control
                velocity = compute_velocity(desired_x, self.stiffness, self.damping)

                # Publish to robot
                twist_msg = Twist()
                twist_msg.linear.x = velocity
                twist_msg.angular.z = 0.0  # only forward/back for now
                self.cmd_pub.publish(twist_msg)

                # Publish placeholder haptic feedback
                self.haptic_pub.publish(Float32(data=abs(velocity)))

                # Log data (timestamp, desired, velocity)
                log_command(desired_x, velocity)

            rate.sleep()


if __name__ == '__main__':
    try:
        TeleopNode()
    except rospy.ROSInterruptException:
        pass
