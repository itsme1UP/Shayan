last_value = 0.0

def map_hand_to_robot(hand_msg, scale=0.5, smooth=0.2):
    """
    Maps Leap Motion hand X position to UR10e forward/backward displacement.
    Includes basic smoothing to reduce jitter.
    """
    global last_value
    try:
        hand_x = hand_msg.axes[0]  # Assume axes[0] is X-axis
        robot_x = hand_x * scale

        # Simple low-pass smoothing
        robot_x = last_value * (1 - smooth) + robot_x * smooth
        last_value = robot_x

        return robot_x
    except Exception as e:
        print("Error mapping hand:", e)
        return last_value
