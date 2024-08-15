### This file will be used to determine the pose needed to angle the bot.
from robot import DiffDriveRobot as robot
import numpy as np
import matplotlib.pyplot as plt

def findObjectCameraFramePosition():
    u = 0
    v = 0
    return u,v

def findObjectRobotRelativePosition(u,v):
    [i,j,_] = np.linalg.inv(robot.H)@[u,v,1]
    return i,j

def findObjectGlobalPosition(i,j):
    x = robot.x + i * np.cos(robot.th)
    y = robot.y + j * np.sin(robot.th)
    return x,y

def object_location(x,y):
    # use to estimate and hope we are correct lol
    plt.figure(figsize=(5,5))
    plt.plot(robot.x,robot.y,'ko')
    plt.plot(x,y,'go')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 10)
    plt.ylim(0, 5)
    plt.show()

def navigate():
    u,v = findObjectCameraFramePosition()
    i,j = findObjectRobotRelativePosition(u,v)
    x,y = findObjectGlobalPosition(i,j)
    th = np.arctan2(y-robot.y,x-robot.x) - robot.th
    # rotate th
    # robot.th = th
    dist = np.sqrt((y-robot.y)**2 + (x-robot.x)**2)
    # drive dist
    # robot.x = x, robot.y = y