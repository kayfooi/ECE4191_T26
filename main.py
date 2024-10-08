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
BALL_CAPACITY = 0 # 5
DEBUG = True # output a lot of images / logs
MOTOR_STOP_CODES = [
    "Reached encoder count",
    "Paddle IR Sensor triggered",
    "Reverse IR Sensor triggered",
    "Timeout",
    "Encoders stopped counting"
]

# Initialise world with relevant quadrant number
W = World(4)

# Initialise robot object
R = Robot(W.getInitPos(), W.getInitHeading())

# Simulation
plt.figure(figsize=(5, 7))
frame_count = 0

def get_save_name(label):
    return f'./main_out/{datetime.now().strftime("%H%M%S")}_{frame_count:04d}_{label}.jpg'

def plot_state(msg=""):
    """
    Save an image of the world from the program's perspective
    For testing and debugging purposes
    """
    global frame_count
    print(msg)
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
    plt.savefig(get_save_name('state'))
    
    frame_count += 1

if R.is_on_pi():
    sim_frame, line_frame, box_frame = None, None, None
else:
    img_size = (640, 480)
    sim_frame = cv2.resize(cv2.imread('CV/test_imgs/test_images/testing0000.jpg'), img_size)
    line_frame = cv2.resize(cv2.imread('CV/test_imgs/test_images/testing0192.jpg'), img_size)
    box_frame = cv2.resize(cv2.imread('CV/test_imgs/box/0004.jpg'), img_size)

# Exploration parameters
consecutive_rotations = 0
rotation_increment = 40
vp_idx = 0
collected_balls = 0


plot_state("Initial state")

# Initial crawl forward (since we cannot observe directly infront of us)
stop_code = R.translate(0.8, speed = 0.15)
if stop_code == 1: # ball detected
    R.collect_ball()
    collected_balls += 1

plot_state("Crawl forward")

while W.getElapsedTime() < COMPETITION_DURATION: 
    # 1. Identify and locate tennis balls
    if DEBUG:
        balls, line_detect_img, YOLO_img = R.detectBalls(sim_frame, visualise=True)

        # Image outputs for debugging
        cv2.imwrite(get_save_name('ball_detect'), line_detect_img)
        cv2.imwrite(get_save_name('YOLO'), YOLO_img)
    else:
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
        print(f"Target ball at: ({target[0], target[1]})")
        # Face ball
        to_face_ball = R.calculateRotationDelta(target)
        print(f"Rotating {to_face_ball:.2f}deg to face ball")
        R.rotate(to_face_ball)

        # Double check existence of ball
        if DEBUG:
            balls, line_detect_img, YOLO_img = R.detectBalls(sim_frame, visualise=True)

            # Image outputs for debugging
            cv2.imwrite(get_save_name('ball_detect_double_check'), line_detect_img)
            cv2.imwrite(get_save_name('YOLO_double_check'), YOLO_img)
        else:
            balls = R.detectBalls(sim_frame)

        found = 0
        for b in balls:
          found += W.addBall(b, R.pos)
        print(f"Found {found} duplicate balls out of {len(balls)} total balls in double check.")

        plot_state("Double check")
        target_checked, target_checked_idx = W.getClosestBall(R.pos)
        rotation = R.calculateRotationDelta(target_checked)
        
        # Facing the ball (within X degrees)
        if abs(rotation) < 2:
            # Travel 99% of the distance to the ball
            r_stop_code, stop_code = R.travelTo(target_checked, 10, 0.15, 0.99)

            if stop_code == 0: # no ball found - rotate side to side
                stop_code = R.rotate(5)
            
            if stop_code == 0:
                stop_code = R.rotate(-5)
            
            if stop_code == 0:
                stop_code = R.translate(0.2, 0.15)
            
            # Detected ball with IR sensor
            if stop_code==1:
                W.removedTarget() # save to world state
                collected_balls += 1
                R.collect_ball()
                plot_state("Collected Ball")
            else:
                plot_state(f"Did not detect ball with IR Sensor. Stop code: {MOTOR_STOP_CODES[stop_code]}")
                W.removedTarget() # remove from state to avoid confusion
    
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
    
    # Navigate to box
    if collected_balls == BALL_CAPACITY or (COMPETITION_DURATION - W.getElapsedTime()) < DUMP_TIME:
        # 5. Navigate to and reverse up to the box

        # Safe exit
        inp = input("Navigating to box, continue? y/n")
        if inp == 'n':
            break 

        # Travel to center of quadrant for best view
        R.travelTo(W.vantage_points[0])

        # Face the theoretical position of the box (0, 0)
        R.rotate(R.calculateRotationDelta(W.origin))

        
        if DEBUG:
            target, res_image = R.detect_box(box_frame, visualise=True)
            cv2.imwrite(get_save_name('box_detect'), res_image)
        else:
            R.detect_box(box_frame)
        
        target = W.box_corner # for simulation [TO REMOVE WHEN TESTING]

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
    
    # Safe exit
    inp = input("Continue? y/n")
    if inp == 'n':
        break

        

        
        

    


