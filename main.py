"""
High level main file to handle ALL logic relevant to the final competion

Algorithm Outline
1. Identify and locate tennis balls
2. Navigate to ball (bug algorithm) or loop through vantage points if no ball found (see world.py)
3. Collect ball
4. Repeat 1 - 3 until we have a full load
     - interrupt ball collection loop if close to the finish time
5. Navigate to and reverse up to the box
6. Dump balls and re-calibrate location/rotation
7. Repeat 1 - 6 until time has elapsed 

Achieve this through a loop that:
- Update the state of the World - position, time elapsed, balls etc.
- Make a decision based on the current state
- Feed instruction to robot
"""

from world import World
from robot import Robot

COMPETITION_DURATION = 60*5 # seconds
DUMP_TIME = 60 # seconds remaining to dump balls

# Initialise world with relevant quadrant number
W = World(4)

# Initialise robot object
R = Robot(W.getInitPos(), W.getInitHeading)

# Exploration parameters
consecutive_rotations = 0
rotation_increment = 40
vp_idx = 0
collected_balls = 0

while W.getElapsedTime() < COMPETITION_DURATION:
    # 1. Identify and locate tennis balls
    balls = R.detectBalls()

    if len(balls) > 0:
        # Add balls to world state
        for b in balls:
            W.addBall(b)
    
    # 2. Navigate to ball (bug algorithm) or loop through vantage points if no ball found
    target = W.getClosestBall(R)
    if target is not None:
        # Face ball
        R.rotate(R.calculateRotationDelta(target))

        # Double check existence of ball
        balls = R.detectBalls()
        for b in balls:
            W.addBall(b)
        target_checked = W.getClosestBall(R)
        rotation = R.calculateRotationDelta(target_checked)
        
        # Facing the ball (within 10 degrees)
        if abs(rotation) < 10:
            # Travel 90% of the distance to the ball
            R.travelTo(target_checked, 10, 0.3, 0.9)

            # TODO: 3. Collect ball
            collected_balls += 1
    else:
        # Decide which direction to rotate on the spot
        if consecutive_rotations == 0:
            if W.is_rotation_sensible(rotation_increment, R):
                rotation_direction = 1
            if W.is_rotation_sensible(-rotation_increment, R):
                rotation_direction = -1
            else:    
                vp_idx = (vp_idx + 1) % len(W.vantage_points)
                R.travelTo(W.vantage_points[vp_idx])

        # Rotate on the spot
        if consecutive_rotations * rotation_increment < 360 and W.is_rotation_sensible(rotation, R):
            R.rotate(rotation_increment)
            consecutive_rotations += 1
        # or move to new vantage point
        else:
            vp_idx = (vp_idx + 1) % len(W.vantage_points)
            R.travelTo(W.vantage_points[vp_idx])
    
    if collected_balls == 4 or (COMPETITION_DURATION - W.getElapsedTime()) < DUMP_TIME:
        # 5. TODO: Navigate to and reverse up to the box
        ...

        # 6. TODO: Dump balls and re-calibrate location/rotation
        ...
        
        

    


