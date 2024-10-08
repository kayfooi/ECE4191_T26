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
from datetime import datetime
import numpy as np

et = time.time()
print(f"Modules loaded in {(et-st) : .3f} sec")

COMPETITION_DURATION = 60*5 # seconds
DUMP_TIME = 60 # seconds remaining to dump balls
BALL_CAPACITY = 1

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
    W.plot_box_park(plt)
    R.plot_bot(plt)

    # Annotate
    plt.title(f"{msg}, Frame: {frame_count}, Time Elapsed: {W.getElapsedTime():.2f}")
    plt.xlabel(f'Bot @ ({R.pos[0]:.2f}, {R.pos[1]:.2f}) facing {R.th:.2f}°')
    plt.legend(loc='upper left')
    plt.axis('equal')
    plt.savefig(f'./main_out/{datetime.now().strftime("%H%M%S")}_{frame_count:04d}.jpg')
    
    frame_count += 1

sim_frame = None if R.is_on_pi() else cv2.resize(cv2.imread('CV/test_imgs/test_images/testing0000.jpg'), (640, 480))
line_frame = None if R.is_on_pi() else cv2.resize(cv2.imread('CV/test_imgs/test_images/testing0192.jpg'), (640, 480))
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
            W.collectedTarget()
            collected_balls += 1
            plot_state("Collected Ball")

            # Not testing paddle
            input("Put the ball in the basket. Press ENTER")
            # R.collect_ball()
        
        
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
        # 5. Navigate to and reverse up to the box

        # Travel to center of quadrant for best view
        R.travelTo(W.vantage_points[0])

        # Face the theoretical position of the box (0, 0)
        R.rotate(R.calculateRotationDelta(W.origin))

        target = W.box_corner # for simulation # R.detect_box()

        plot_state("Translated to center and faced box")

        if target is not None:

            # Face box
            R.rotate(R.calculateRotationDelta(target))

            # Travel to closer to box if far away
            distance_to_box = R.calculateDistance(target)
            if R.calculateDistance(target) > 1.0:
                R.travelTo(target, complete=0.6)

            # orient bot with line
            distance_to_line = None
            consecutive_rotations = 0

            # face the y = 0 line
            to_face_line = R.calculateRotationDelta(np.array([0, R.pos[1]]))
            R.rotate(to_face_line) # rotate to face a line
            plot_state("Rotation facing line")
            distance_to_line = 1.25 # R.get_perpendicular_to_line(distance=abs(R.pos[1])+1)
            
            # No line found. Intervention needed
            if distance_to_line is None:
                print("I am lost. Put me in-front to the box and I'll dump the balls.")
                input("Press ENTER when done.")
                # R.dump_balls()
                
            else:
                # stop early
                R.translate(distance_to_line-0.15, speed=0.2)
                to_face_box = round(R.calculateRotationDelta(W.origin)/90) * 90
                R.rotate(-to_face_box) # face away from box

                # Reverse to box (make this a function in robot or WheelMotor)
                R.translate(-(distance_to_box**2 - distance_to_line**2)**0.5)

                # if successful
                # R.dump_balls()
            
            # reset location
            R.pos = W.box_park.copy()
            # reset orientation
            R.th = 90 if R.pos[1] > 0 else -90

            plot_state("Reset location")

            collected_balls = 0
        

        
        

    


