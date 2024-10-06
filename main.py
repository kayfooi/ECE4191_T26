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
et = time.time()
print(f"main.py: Modules loaded in {(et-st) : .3f} sec")

COMPETITION_DURATION = 60*5 # seconds
DUMP_TIME = 60 # seconds remaining to dump balls
BALL_CAPACITY = 4
ROTATION_INCREMENT = 40 # degrees (exploration)

# Initialise world with relevant quadrant number
W = World(4)

# Initialise robot object
R = Robot(W.getInitPos(), W.getInitHeading())

# Simulation frame tracker
frame_count = 0

def main():
    # image to feed to simulation
    sim_frame = None if R.is_on_pi() else cv2.imread('CV/test_imgs/test_images/testing0000.jpg')

    # Exploration variables
    consecutive_rotations = 0
    vp_idx = 0
    collected_balls = 0

    plot_state("Initial state")

while W.getElapsedTime() < COMPETITION_DURATION: 
    # 1. Identify and locate tennis balls
    balls = R.detectBalls(sim_frame)

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
            # balls = R.detectBalls(sim_frame)
            # for b in balls:
            #     W.addBall(b)
            
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
            R.collect_ball()
        
        
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
            if consecutive_rotations * ROTATION_INCREMENT < 360 and W.is_rotation_sensible(ROTATION_INCREMENT, R):
                R.rotate(ROTATION_INCREMENT)
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

                R.rotate(180)

                plot_state("Moved close to box")

                # TODO new drive function that makes the bot go straight until the infared sensor goes off 
                
                
                W.collectTarget()
                R.dump_balls()

                plot_state("Deposited Balls")
                # TODO drive away from box ( if we have time itd be cool )
        
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

        # boxPosition = determineBoxPosition(box)
        # R.travelTo(boxPosition)
        # R.rotate(180)

        # 6. TODO: Dump balls and re-calibrate location/rotation
        

        
        

    


