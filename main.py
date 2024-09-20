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
import time

st = time.time()
from world import World
from robot import Robot
from matplotlib import pyplot as plt
import sys
import cv2

et = time.time()
print(f"Modules loaded in {(et-st) : .3f} sec")

COMPETITION_DURATION = 60*5 # seconds
DUMP_TIME = 60 # seconds remaining to dump balls
BALL_CAPACITY = 4

# Initialise world with relevant quadrant number
W = World(4)

# Initialise robot object
R = Robot(W.getInitPos(), W.getInitHeading())

# Simulation
plt.figure(figsize=(5, 7))
frame_count = 0
def plot_state(msg=""):
    """
    Save an image of the world from the program's perspective
    For testing and debugging purposes
    """
    global frame_count
    # clear figure
    plt.clf()

    # plot state
    W.plot_court_lines(plt)
    W.plot_vps(plt)
    W.plot_balls(plt)
    R.plot_bot(plt)

    # Annotate
    plt.title(f"{msg}, Frame: {frame_count}, Time Elapsed: {W.getElapsedTime():.2f}")
    plt.xlabel(f'Bot @ ({R.pos[0]:.2f}, {R.pos[1]:.2f}) facing {R.th:.2f}°')
    plt.legend(loc='upper left')
    plt.axis('equal')
    plt.savefig(f'./main_out/{frame_count:03d}.jpg')
    
    frame_count += 1

sim_frame = None if R.is_on_pi() else cv2.imread('CV/test_imgs/test_images/testing0042.jpg')

# Exploration parameters
consecutive_rotations = 0
rotation_increment = 40
vp_idx = 0
collected_balls = 0

plot_state("Initial state")

while W.getElapsedTime() < COMPETITION_DURATION:
    # 1. Identify and locate tennis balls
    balls = R.detectBalls(sim_frame)

    if len(balls) > 0:
        # Add balls to world state
        for b in balls:
            W.addBall(b, R.pos)
    
    
    
    # 2. Navigate to ball (bug algorithm) or loop through vantage points if no ball found
    target, target_idx = W.getClosestBall(R.pos)
    W.target_ball_idx = target_idx

    plot_state("First detection")
    

    if target is not None:
        # Face ball
        R.rotate(R.calculateRotationDelta(target))

        plot_state("Rotation")

        # Double check existence of ball
        # balls = R.detectBalls()
        # for b in balls:
        #   W.addBall(b)
        
        target_checked, target_checked_idx = W.getClosestBall(R.pos)
        rotation = R.calculateRotationDelta(target_checked)
        
        # Facing the ball (within X degrees)
        if abs(rotation) < 5:
            # Travel 99% of the distance to the ball
            R.travelTo(target_checked, 10, 0.3, 0.99)

            plot_state("Moved close to ball")
            # TODO: 3. Collect ball
            W.collectTarget()
            collected_balls += 1
            plot_state("Collected Ball")
        
        
    else:
        # Decide which direction to rotate on the spot
        if consecutive_rotations == 0:
            if W.is_rotation_sensible(rotation_increment, R):
                rotation_direction = 1
            elif W.is_rotation_sensible(-rotation_increment, R):
                rotation_direction = -1
            else:    
                vp_idx = (vp_idx + 1) % len(W.vantage_points)
                R.travelTo(W.vantage_points[vp_idx])

        # Rotate on the spot
        if consecutive_rotations * rotation_increment < 360 and W.is_rotation_sensible(rotation_increment, R):
            R.rotate(rotation_increment)
            consecutive_rotations += 1
            plot_state("Rotation because no balls found")
        # or move to new vantage point
        else:
            vp_idx = (vp_idx + 1) % len(W.vantage_points)
            R.travelTo(W.vantage_points[vp_idx])
            plot_state("Translation because no balls found")

    if frame_count > 20:
        sys.exit()
    
    # Navigate to box
    if collected_balls == BALL_CAPACITY or (COMPETITION_DURATION - W.getElapsedTime()) < DUMP_TIME:
        # 5. TODO: Navigate to and reverse up to the box
        ...

        # 6. TODO: Dump balls and re-calibrate location/rotation
        ...
        
        

    


