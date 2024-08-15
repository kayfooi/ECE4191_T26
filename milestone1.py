"""
Main file to complete milestone one:
1. Identify and locate tennis balls
2. Navigate to ball (bug algorithm)
3. Return to starting position, just reverse
   the motors

"""
from robot import DiffDriveRobot


if __name__ == "__main__":
    bot = DiffDriveRobot()

    # Initial Scan (runs in the background)
    bot.rotate(360)

    detected = None
    while not None:
        detected = bot.detect_ball()



