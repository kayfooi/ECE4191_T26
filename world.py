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

class Ball:
    def __init__(self, id, x, y):
        self.id = id
        self.pos = np.array(x, y)
        self.collected = False
