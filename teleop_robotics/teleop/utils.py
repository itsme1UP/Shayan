import csv
import os
import time

LOG_FILE = 'teleop_log.csv'

# Make sure log file exists and has headers
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'desired_x', 'velocity', 'haptic'])


def log_command(desired_x, velocity, haptic=0.0):
    """Logs the command and feedback to a CSV file"""
    try:
        timestamp = time.time()
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, desired_x, velocity, haptic])
    except Exception as e:
        print('Logging error:', e)


def clamp(value, min_val, max_val):
    """Clamps a value between min and max"""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value
