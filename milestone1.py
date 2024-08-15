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

if __name__ == "__main__":
   init_pos = np.array([0,0])
   init_orientation = 0
   bot = DiffDriveRobot(init_pos, init_orientation)

   # 1. Identify and locate exactly 1 tennis ball
   detected = None
   while detected is None:
      detected = bot.detect_ball() # scan environment
      if (detected is not None):
         # double check its a ball
         
         # rotate to face ball
         rotation = bot.calculateRotationDelta(detected)
         bot.rotate(rotation)

         # attempt to detect ball again
         detected = bot.detect_ball()
         if (detected is not None):
            rotation = bot.calculateRotationDelta(detected)
            if abs(rotation) < 5:
               # successfully detected ball again
               # (and it is in front of robot +- 5 degrees)
               break

      # Continuously rotate in fixed increments until bot finds a ball       
      bot.rotate(40)
   
   # 2. Navigate to ball
   # initial distance to ball
   init_distance = bot.calculateDistance(detected)
   distance = init_distance

   # repeatedly halve distance and correct rotation until in close range of ball
   while distance > 0.8:
      bot.translate(distance / 2)
      detected = bot.detect_ball()
      rotation = bot.calculateRotationDelta(detected)
      distance = bot.calculateDistance(detected)
      # rotate if not on the right path
      if abs(rotation) > 5:  
         bot.rotate(rotation)
   
   # slowly approach the ball (and hopefully touch it)
   bot.translate(distance + 0.05, 0.1)


   # 3. Return to starting position
   home_distance = bot.calculateDistance(init_pos)
   home_rotation = bot.calculateRotationDelta(init_pos)

   # reverse home
   # If all goes to plan we should be facing in a direction
   # directly away from our starting point. This rotation
   # makes sure of it.
   bot.rotate((home_rotation + 180) % 360)
   bot.translate(-home_distance)
      
   
   






   



