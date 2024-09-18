"""
Represents the map of the court
"""

import numpy as np
import time
import unittest

QUAD_WIDTH = 4.11 # width of all quadrant
NETLINE = 6.40 # y distance from net to center line
BASELINE = 5.48 # y distance from baseline to center line
SP_OS = 0.2 # Start Point Offset: Start the robot at this offset from corner in both x and y directions

QUADRANTS = [
    {
        "name": "Quadrant 1",
        "xrange": [0.00, QUAD_WIDTH],
        "yrange": [0.00, NETLINE],
        "init_pos": [QUAD_WIDTH - SP_OS, NETLINE - SP_OS],
        "init_heading": -90
    },
    {
        "name": "Quadrant 2",
        "xrange": [-QUAD_WIDTH, 0.00],
        "yrange": [0.00, NETLINE],
        "init_pos": [-QUAD_WIDTH + SP_OS, NETLINE - SP_OS],
        "init_heading": -90
    },
    {
        "name": "Quadrant 3",
        "xrange": [-QUAD_WIDTH, 0.00],
        "yrange": [-BASELINE, 0.00],
        "init_pos": [-QUAD_WIDTH+SP_OS, -BASELINE+SP_OS],
        "init_heading": 90
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
    """
    Holds the current state of the world. Robot acts on this state.
    """
    def __init__(self, quadrant:int):
        self.balls = []
        self.quadrant = QUADRANTS[quadrant-1]
        self.init_time = time.time()

        # Generating vantage Points
        # (Points that the robot will go to to look for balls if it hasn't found any)
        center = np.array([ np.sum(self.quadrant["xrange"])/2, 
                            np.sum(self.quadrant["yrange"])/2])
        os = (center - np.array(self.quadrant["init_pos"])) * 0.55
        self.vantage_points = center + np.array([
            [0, 0],
            [1, 1],
            [-1, 1],
            [0, 0],
            [1, -1],
            [-1, -1]
        ]) * os

    def getInitPos(self):
        return np.array(self.quadrant["init_pos"])

    def getInitHeading(self):
        return np.array(self.quadrant["init_heading"])

    def getElapsedTime(self):
        return time.time() - self.init_time

    def getDistancesToBalls(self, loc):
        return np.linalg.norm(np.array(self.balls) - loc, axis=1).reshape((-1))

    def getClosestBall(self, pos):
        if len(self.balls) > 0:
            min_idx = np.argmin(self.getDistancesToBalls(pos))
            return self.balls[min_idx]
        else:
            return None

    def addBall(self, ball, pos):
        """
        Record a ball's location in the world state

        Parameters
        ---
        ball: ArrayLike
            2D world coordinate of ball
        
        pos: ArrayLike
            2D world coordinate of bot
        
        Returns
        ---
        duplicateFound: bool
            Whether the ball has been seen before
        """
        if len(self.balls) == 0:
            self.balls.append(ball)
            return False # no duplicate found
        
        # threshold dependent on distance to ball
        duplicate_threshold = min(0.05, np.linalg.norm(ball-pos) * 0.05)

        # distance between all detected balls and new ball
        distances = self.getDistancesToBalls(ball)

        mindist = np.argmin(distances)
        if distances[mindist] < duplicate_threshold:
            self.balls[mindist] = ball # replace duplciate ball
            return True # duplicate found
        else:
            self.balls.append(ball) # add new ball
            return False # no duplicate found
    
    def is_rotation_sensible(self, r, b):
        """
        Check if the rotation will yield good visibilty of our quadrant
        """
        r = np.radians(r + b.th)
        thresh = 1.0 # check this many meters in front of the robot
        point = b.pos + thresh * np.array([np.cos(r), np.sin(r)])
        return self.is_point_in_quad(point)

    
    def is_point_in_quad(self, p):
        xr = self.quadrant["xrange"]
        yr = self.quadrant["yrange"]
        return xr[0] < p[0] < xr[1] and yr[0] < p[1] < yr[1]

    def plot_court_lines(self, ax):
        """
        Plots tennis court lines onto ax
        """
        for hline in [NETLINE, 0, -BASELINE]:
            ax.plot([-QUAD_WIDTH, QUAD_WIDTH], [hline]*2, 'k-', label=None, linewidth=1.5)
        for vline in [-QUAD_WIDTH, 0, QUAD_WIDTH]:
            ax.plot([vline]*2, [NETLINE, -BASELINE], 'k-', label=None, linewidth=1.5)
    
    def plot_vps(self, ax, c='k'):
        """
        Plots vantage points onto ax
        """
        vps = w.vantage_points
        plt.plot(vps[:, 0], vps[:, 1], marker='x', linestyle='None', label=f'Quadrant {q}', color=c)


class TestBot(unittest.TestCase):
    """
    Test Camera specific functions
    """
    def setUp(self):
        self.W = World(4)

    def test_balls(self):
        # addBall returns true if duplicate is found, false otherwise
        self.assertFalse(self.W.addBall(np.array([1, 1]), np.array([0, 0])))
        self.assertFalse(self.W.addBall(np.array([2, 2]), np.array([0, 0])))
        self.assertTrue(self.W.addBall(np.array([2.05, 2]), np.array([0, 0])))
        self.assertFalse(self.W.addBall(np.array([2.11, 2]), np.array([1.5, 2])))
        self.assertEqual(len(self.W.balls), 3)

        # test closest ball
        np.testing.assert_array_equal(self.W.balls[0], self.W.getClosestBall(np.array((0.5, 0.5))))
        np.testing.assert_array_equal(self.W.balls[2], self.W.getClosestBall(np.array((2.10, 2))))

    def test_is_ball_in_quad(self):
        q1 = World(1)
        q2 = World(2)
        q3 = World(3)
        q4 = World(4)

        self.assertTrue(q1.is_point_in_quad(np.array((1, 1))))
        self.assertTrue(q2.is_point_in_quad(np.array((-1, 1))))
        self.assertTrue(q3.is_point_in_quad(np.array((-1, -1))))
        self.assertTrue(q4.is_point_in_quad(np.array((1, -1))))

        self.assertFalse(q1.is_point_in_quad(np.array((-1, 1))))
        self.assertFalse(q2.is_point_in_quad(np.array((1, 1))))
        self.assertFalse(q3.is_point_in_quad(np.array((1, -1))))
        self.assertFalse(q4.is_point_in_quad(np.array((-1, -1))))


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    # Show intitial world state
    cols = ['r', 'g', 'm', 'b']
    arrow_size = 0.5
    for q in range(1,5):
        w = World(q)
        c = cols[q-1]
        
        # Plot vantage points
        w.plot_vps(plt, c)

        # Plot inital heading/position
        ip = w.getInitPos()
        ih = np.radians(w.getInitHeading())

        plt.arrow(ip[0], ip[1], arrow_size*np.cos(ih), arrow_size*np.sin(ih), color=c, width=0.1)
    
    plt.title('Court dimensions, vantage points and initial robot state stored in world.py')
    
    w.plot_court_lines(plt)
    
    plt.axis('equal')
    plt.legend()
    # plt.show()

    unittest.main()

