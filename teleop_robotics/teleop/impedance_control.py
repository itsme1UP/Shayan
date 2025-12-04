last_velocity = 0.0

def compute_velocity(desired_x, stiffness=50.0, damping=20.0, current_x=0.0, dt=0.05, max_vel=1.0):
    """
    Compute robot velocity using simple impedance control:
    F = K*(xd - x) + D*(vd - v)
    For simplicity, map force to velocity directly.
    Includes basic clamping and smoothing.
    """
    global last_velocity
    try:
        # Position error
        error = desired_x - current_x

        # Desired velocity based on impedance
        vel = stiffness * error - damping * last_velocity

        # Simple smoothing to reduce jitter
        alpha = 0.2
        vel = last_velocity * (1 - alpha) + vel * alpha

        # Clamp to max velocity
        if vel > max_vel:
            vel = max_vel
        elif vel < -max_vel:
            vel = -max_vel

        last_velocity = vel
        return vel

    except Exception as e:
        print("Error in impedance control:", e)
        return last_velocity
