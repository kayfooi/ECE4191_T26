"""
Represents the map of the court
"""

import numpy as np


QUAD_WIDTH = 4.11 # width of all quadrant
QUAD_HEIGHT = 6.40 # height of quads closest to net
BASELINE = 5.48 # y distance from baseline to quad 1 or 2
SP_OS = 0.2 # Start Point Offset: Start the robot at this offset from corner in both x and y directions

QUADRANTS = [
    {
        "name": "Quadrant 1",
        "xrange": [0.00, QUAD_WIDTH],
        "yrange": [0.00, QUAD_HEIGHT],
        "init_pos": [QUAD_WIDTH - SP_OS, QUAD_HEIGHT - SP_OS],
        "init_heading": -90
    },
    {
        "name": "Quadrant 2",
        "xrange": [-QUAD_WIDTH, 0.00],
        "yrange": [0.00, QUAD_HEIGHT],
        "init_pos": [-QUAD_WIDTH + SP_OS, QUAD_HEIGHT - SP_OS],
        "init_heading": -90
    },
    {
        "name": "Quadrant 3",
        "xrange": [-QUAD_WIDTH, 0.00],
        "yrange": [-BASELINE, 0.00],
        "init_pos": [-SP_OS, -BASELINE+SP_OS],
        "init_heading": 180
    },
    {
        "name": "Quadrant 4",
        "xrange": [0.00, QUAD_WIDTH],
        "yrange": [-BASELINE, 0.00],
        "init_pos": [QUAD_WIDTH-SP_OS, -BASELINE+SP_OS],
        "init_heading": 90
    }
]
    



class World:
    def __init__(self, quadrant:int):
        self.balls = []
        self.quadrant = QUADRANTS[quadrant-1]
        self.center = np.array(
            [np.sum(self.quadrant["xrange"])/2, 
            np.sum(self.quadrant["yrange"])/2]
            )
        os = (self.center - np.array(self.quadrant["init_pos"])) * 0.67
        self.interest_points = self.center + np.array([
            [0, 0],
            [1, 1],
            [-1, 1],
            [0, 0],
            [1, -1],
            [-1, -1]
        ]) * os

    def getInitPos(self):
        return np.array(self.quadrant["init_pos"])
    
    def is_rotation_sensible(self, r, b):
        r = np.radians(r + b.th)
        thresh = 1.0 # check this many meters in front of the robot
        point = b.pos + thresh * np.array([np.cos(r), np.sin(r)])
        return self.is_point_in_quad(point)
    
    def is_point_in_quad(self, p):
        xr = self.quadrant["xrange"]
        yr = self.quadrant["yrange"]
        return xr[0] < p[0] < xr[1] and yr[0] < p[1] < yr[1]


class Ball:
    def __init__(self, id, x, y):
        self.id = id
        self.pos = np.array(x, y)
        self.collected = False


if __name__ == "__main__":
    # from matplotlib import pyplot as plt
    w = World(4)
    ips = np.array(w.interest_points)
    print(ips)
    # plt.plot(ips[:, 0], ips[:, 1], 'rx')
    # plt.title(f'{w.quadrant["name"]} Interest Points')
    # plt.show()