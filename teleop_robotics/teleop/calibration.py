import yaml

# Simple calibration for mapping Leap Motion hand positions to UR10e workspace

class Calibration:
    def __init__(self, config_file='config/params.yaml'):
        # Load saved offsets and scale from config
        self.offset = 0.0
        self.scale = 1.0
        try:
            with open(config_file, 'r') as f:
                params = yaml.safe_load(f)
                self.offset = params.get('hand_offset', 0.0)
                self.scale = params.get('hand_scale', 0.5)
        except Exception as e:
            print('Error loading calibration config:', e)

    def calibrate_hand(self, hand_pos):
        # Apply offset and scale
        try:
            robot_pos = (hand_pos - self.offset) * self.scale
            return robot_pos
        except Exception as e:
            print('Calibration error:', e)
            return 0.0

if __name__ == '__main__':
    # Simple CLI test
    calib = Calibration()
    while True:
        inp = input('Enter hand X position (-1 to 1): ')
        try:
            val = float(inp)
            print('Mapped robot X:', calib.calibrate_hand(val))
        except:
            print('Invalid input')
