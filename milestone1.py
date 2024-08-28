"""
NOTES

High level main file to complete milestone one:
1. Identify and locate exactly 1 tennis ball
2. Navigate to ball (bug algorithm)
3. Return to starting position, just reverse
   the motors

Currently thinking the rotate and translate functions should be
blocking because its a lot easier to debug and run compared
to using asynchronous calls - Joseph

Purely vision based for the moment
"""
from robot import DiffDriveRobot
import numpy as np
import cv2
import world

if __name__ == "__main__":
   
   ROTATION_INCREMENT = 40 # amount to rotate each time in degrees
   
   court = world.World(3) # either quadrant 3 or 4 for milestone 1
   # bot = DiffDriveRobot(court.quadrant["init_pos"], court.quadrant["init_heading"])
   bot = DiffDriveRobot([0, 0], 0)
   # test_img = cv2.imread('./CV/test_imgs/blender/simple/test_0.jpg')
   
   # 1. Identify and locate exactly 1 tennis ball
   detected = None
   consecutive_rotations = 0
   max_consecutive_rotations = np.ceil(360/ROTATION_INCREMENT)
   current_loc = -1
   while detected is None:
      detected = bot.detect_ball() # scan environment (return the closest ball)
      print("Detected:", detected)
      if (detected is not None):
         # double check its a ball
         
         # # rotate to face ball
         # rotation = bot.calculateRotationDelta(detected)
         # bot.rotate(rotation)
         bot.coordTran(detected)

         # bot.cap.release()

         # attempt to detect ball again
         # detected = bot.detect_ball()
         # if (detected is not None):
         #    rotation = bot.calculateRotationDelta(detected)
         #    if abs(rotation) < 20:
         #       # successfully detected ball again
         #       # (and it is in front of robot +- 5 degrees)
         #       break

      # Continuously rotate in fixed increments until bot finds a ball
      # if consecutive_rotations < max_consecutive_rotations:
      #    bot.rotate(ROTATION_INCREMENT)
      #    consecutive_rotations += 1

      # Translate to a better location
      # else:
      #    current_loc = (current_loc + 1) % len(court.interest_points)
      #    new_loc = court.interest_points[current_loc]
         
      #    rotation = bot.calculateRotationDelta(new_loc)
      #    distance = bot.calculateDistance(new_loc)
      #    print(f'Going to {current_loc}: ({new_loc[0]}, {new_loc[1]}) w/ rotation: {rotation} and translation: {distance}')

      #    bot.rotate(rotation)
      #    bot.translate(distance)
      
   
   # # 2. Navigate to ball
   # # initial distance to ball
   # init_distance = bot.calculateDistance(detected)
   # distance = init_distance

   # # repeatedly halve distance and correct rotation until in close range of ball
   # while distance > 0.8:
   #    bot.translate(0.005)
   #    detected = bot.detect_ball()
   #    rotation = bot.calculateRotationDelta(detected)
   #    distance = bot.calculateDistance(detected)
   #    # rotate if not on the right path
   #    if abs(rotation) > 5:  
   #       bot.rotate(rotation)
   
   # # slowly approach the ball (and hopefully touch it)
   # bot.translate(distance + 0.05, 0.1)


   # 3. Return to starting position
   # home_distance = bot.calculateDistance(init_pos)
   # home_rotation = bot.calculateRotationDelta(init_pos)

   # # reverse home
   # # If all goes to plan we should be facing in a direction
   # # directly away from our starting point. This rotation
   # # makes sure of it.
   # bot.rotate((home_rotation + 180) % 360)
   # bot.translate(-home_distance)
      
   
   






   



