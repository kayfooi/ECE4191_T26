import numpy as np
import unittest
import cv2
import time
from camera import Camera
import io
from datetime import datetime

try:
    import pigpio
    from WheelMotor import DiffDrive
    on_pi = True
except ImportError:
    print("pigpiod library not found. Continuing tests with no pi")
    on_pi = False

if on_pi:
    from rPi_sensor import laser
    # tof = laser.PiicoDev_VL53L1X()
    from servo import Servo
else:
    tof = None

# GPIO Numbers
TIPPING_SERVO_GPIO = 7
PADDLE_SERVO_GPIO = 14
# Other GPIO is stored in WheelMotor.py

class Robot:
    def __init__(self,init_pos = np.array([0.0, 0.0]), init_th = 0):
        self.pos = init_pos
        self.th = np.float64(init_th)
        if on_pi:
            self.pi = pigpio.pi()
            self.dd = DiffDrive(self.pi)
            
            self.tip_servo = Servo(self.pi, TIPPING_SERVO_GPIO, 150) 
        
            self.paddle_servo = Servo(self.pi, PADDLE_SERVO_GPIO, 180)
        else:
            self.pi = None 
            self.dd = None 
            self.tip_servo = None 
            self.paddle_servo = None
        
        self.camera = Camera(open_cam=True)
    
    def is_on_pi(self):
        return on_pi
    
    # -------- CONTROL FUNCTIONS --------
    def rotate(self, angle, speed=20):
        """
        Rotate anti-clockwise  
        
        Parameters
        ----
        angle: Amount of rotation in degrees, can be negative. Should be in range (-180, +180)
        speed: rotational speed in degrees per second
        """
        # Send rotation instruction
        if on_pi:
            rotation_left, rotation_right, stop_code = self.dd.rotate(-angle, speed)
            if angle > 0:
                self.th += (abs(rotation_left) + abs(rotation_right)) / 2
            else:
                self.th -= (abs(rotation_left) + abs(rotation_right)) / 2
            return stop_code
        else:
            noise = 2 # magnitude of randomness (simulation)
            self.th += angle + np.random.random() * noise - noise/2
            return 0
        

    
    def translate(self, displacement, speed=0.3):
        """
        Drive the robot in a straight line

        Parameter
        ----
        displacement: Amount to drive in meters, can be negative
        speed: speed in meters per second
        """
        # Send translation instruction
        if on_pi:
            disp_left, disp_right, stop_code = self.dd.translate(displacement, speed)
            avg_disp = (disp_left + disp_right )/ 2
            
            # Big Error
            if abs(avg_disp - displacement) > 0.1:
                print(f"Translation error large: {avg_disp - displacement:.3f} m")
            if abs(avg_disp - disp_left) > 0.05 or abs(avg_disp - disp_right) > 0.05:
                print(f"Large variance between left ({disp_left:.3f} m) and right ({disp_right:.3f} m) motor displacement")

            # Update position coordinates
            th_rad = np.radians(self.th)
            self.pos += np.array([
                avg_disp * np.cos(th_rad),
                avg_disp * np.sin(th_rad)
            ])
            return stop_code
        else:
            th_rad = np.radians(self.th)
            noise = 0.05
            avg_disp = displacement + noise * np.random.random() - noise/2
            self.pos += np.array([
                avg_disp * np.cos(th_rad),
                avg_disp * np.sin(th_rad)
            ])
            return 0 # normal stop_code

    
    def travelTo(self, p, rspeed=20.0, tspeed=0.3, complete=1.0):
        """
        Travel to given point, p, through rotation and translation

        Parameters
        ---
        p: ArrayLike
            2D world coordinate target location
        rspeed: float
            Rotational speed (degrees/sec)
        tspeed: float
            Straight line speed (m/sec)
        complete: float
            Total displacement to complete, ranging from (0.0 to 1.0)
        """
        rotation = self.calculateRotationDelta(p)
        disp = self.calculateDistance(p) * complete
        r_stop_code = self.rotate(rotation)
        t_stop_code = self.translate(disp)
        return r_stop_code, t_stop_code
    
    def collect_ball(self):
        """
        Actuates paddle mechanism to collect ball
        """
        rest_angle = 180 # horizontal paddle
        collect_angle = 0 # paddle aligns with bucket
        collect_speed = 90 # degrees per second (can't be too quick or servo will stall)
        
        # make sure it is at home
        self.paddle_servo.set_angle(rest_angle)

        self.paddle_servo.set_angle(collect_angle)
        time.sleep(1) # allow the ball to roll off
        self.paddle_servo.set_angle(rest_angle, 50)
        time.sleep(1)
        self.paddle_servo.stop()

        # No feedback, may need sensors to somehow detect if ball is collected
        # Hopefully it works reliably enough that we don't need this
    
    def dump_balls(self):
        """
        Actuates tipping mechanism to dump balls
        """
        # Dump
        dump_angle = 60 # 
        dump_speed = 180 # degrees per second

        # set paddle servo out of the way
        self.paddle_servo.set_angle(90, 70)

        # no speed limit - smoother dump
        self.tip_servo.set_angle(dump_angle, dump_speed)

        time.sleep(3) # allow balls to exit
        # TODO: may have to add shaking mechanism if balls don't exit reliably

        # Return to original position
        rest_angle = 150 # rest on the lip
        return_speed = 50


        self.tip_servo.set_angle(rest_angle, return_speed)
        time.sleep(0.5)
        self.tip_servo.stop()

        # return paddle servo to home
        self.paddle_servo.set_angle(180, 50)
    
    def get_perpendicular_to_line(self, distance=2.0, img=None):
        """
        When facing a line, rotate such that the robot's heading is perpendicular to the line
        Use to re-orient the bot

        Returns
        ---
        Distance away from line
        """
        # Image coord
        if img is None:
            img = self.camera.capture()
        
        target_point = self.camera.world_to_image(np.array([[distance, 0, 1]]))[0]
        rotation_needed = 100
        while rotation_needed < 2:
            line_pair = self.camera.detect_lines(tuple(target_point.astype(int)), img)
            if line_pair is not None:
                offset = 30 # in pixels
                m = (line_pair[0][0] + line_pair[1][0]) / 2 # average slope
                c = (line_pair[0][1] + line_pair[1][1]) / 2# average intercept
                x1, x2 = target_point[0] - offset, target_point[0] + offset
                y1, y2 = m*x1 + c, m*x2 + c
                world_coords = self.camera.image_to_world(np.array([
                    [x1, y1],
                    [x2, y2]
                ]))

                # Calculate angle of line in real world (should be a multiple of 90deg)
                dx = world_coords[0, 0] - world_coords[1, 0]
                dy = world_coords[0, 1] - world_coords[1, 1]
                detected_angle = np.degrees(np.arctan2(dy, dx))
                actual_angle = round(detected_angle/90) * 90 # clip to the nearest 90 degree orientation
                rotation_needed = detected_angle - actual_angle
                
                print(f"Rotating {rotation_needed:.2f} deg to face line")
                self.rotate(rotation_needed)
            else:
                return None
        
    
    # -------- HELPER FUNCTIONS -----------
    
    def _getRotationMatrix(self):
        """
        Get 2D rotation matrix from bot orientation
        """
        th_rad = np.radians(self.th)
        return np.array([
            [np.cos(th_rad), -np.sin(th_rad)],
            [np.sin(th_rad), np.cos(th_rad)]
        ])

    def calculateRotationDelta(self, p):
        """
        Parameters
        ----
        p: np.array (x, y) coordinates

        Return
        ----
        Rotation needed by bot in order to face p in degrees
        """
        delta = p - self.pos
        r = (np.degrees(np.arctan2(delta[1], delta[0])) - self.th)
        if r < -180:
            return r + 360
        if r > 180:
            return r - 360
        else:
            return r
    
    def calculateDistance(self, p):
        """
        Parameters
        ----
        p: np.array (x, y) coordinates

        Return
        ----
        Distance between bot and p
        """
        delta = p - self.pos
        return np.sqrt(np.sum(delta ** 2))
    
    def detectBalls(self, img=None, visualise=False):
        """
        Detects ball locations

        Paramters
        ----
        img: ArrayLike
            Optional test image
        
        Return
        ---
        balls_locs ArrayLike
            Array of detected ball locations in 2D world coordinates
        """

        # TODO: when calsibrating camera again, ensure world coordinates are relative to bot facing 0 deg
        
        if visualise:
            relative_pos, line_detect_image, YOLO_img = self.camera.detectBalls(img, visualise=True)
        else:
            relative_pos = self.camera.detectBalls(img)

        if len(relative_pos) > 0:
            ball_locs = relative_pos @ self._getRotationMatrix().T + self.pos
            
        else:
            ball_locs = []
        
        if visualise:
            return ball_locs, line_detect_image, YOLO_img
        else:
            return ball_locs
    
    def detect_box(self, img=None, visualise=False):
        if visualise:
            relative_pos, res_image = self.camera.detect_box(img, visualise=True)
        else:
            relative_pos = self.camera.detect_box(img)

        if len(relative_pos) > 0:
            box_locs = relative_pos @ self._getRotationMatrix().T + self.pos
            # Find closest box to (0, 0) (where it should be)
            distances_from_origin = np.linalg.norm(box_locs, axis=1)
            correct_loc_box_idx = np.argmin(distances_from_origin)

            # Within 1.5 m of origin
            if distances_from_origin[correct_loc_box_idx] < 1.5:
                box_loc = box_locs[correct_loc_box_idx]
            else:
                box_loc = None
        else:
            box_loc = None
        
        if visualise:
            return box_loc, res_image
        else:
            return box_loc

    # -------- VISUALISING FUNCTIONS --------
    def plot_bot(self, ax):
        """
        Show bot on ax
        """
        arrow_size = 0.2
        heading = np.radians(self.th)
        ax.arrow(self.pos[0], self.pos[1], 
                 arrow_size*np.cos(heading), 
                 arrow_size*np.sin(heading), 
                 color='k', width=arrow_size/3.5, label=None)
    
    def __str__(self):
        """
        Print representation of robot
        """
        time_now = datetime.now().strftime("%H:%M:%S")
        pos = f'({self.pos[0]:.3f}, {self.pos[1]:.3f})'
        return f'Robot @ {time_now}: {pos} facing {self.th:.1f} deg'




class TestBot(unittest.TestCase):
    """
    Test Robot specific functions. See `camera.py` for camera testing
    """
    ACTIVE_TESTS = [
        # "left_motor",
        # right_motor",
        # "rotation",
        # "translation",
        # "rotation_translation",
        # "ball_detection_cam"
        # "collect_ball"
        # "dump_balls"
        "collect_and_dump"
        # "detect_travel_collect_dump"
        # "detect_and_travel_to"
        # "get_perpendicular_to_ball"
        #" travel_to"
    ]
    
    def setUp(self):
        self.bot = Robot()

    @unittest.skipIf("rotation_translation" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_rotation_translation(self):
        init_pos = self.bot.pos.copy()

        rotation = -45
        self.bot.rotate(rotation)
        distance = np.sqrt(0.3**2 + 0.3**2)
        self.bot.translate(distance)
        
        np.testing.assert_allclose(self.bot.pos, np.array([0.3, 0.3]), atol=5e-2)
        self.assertLess(abs(self.bot.calculateDistance(init_pos)-distance), 0.1)
        self.assertLess(abs(abs(self.bot.calculateRotationDelta(init_pos))-180), 0.1)
        print(self.bot)
    
    @unittest.skipIf("get_perpendicular_to_ball" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_get_perpendicular_to_ball(self):
        img = cv2.imread('CV/test_imgs/test_images/testing0192.jpg')
        img = cv2.resize(img, (640, 480))
        self.bot.get_perpendicular_to_line(distance = 6.0, img=img)
    
    @unittest.skipIf("travel_to" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_travel_to(self):
        target = np.array([3, 4])
        self.bot.travelTo(target)
        np.testing.assert_allclose(self.bot.pos, target, atol=1e-1)

    @unittest.skipIf("detect_and_travel_to" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_detect_and_travel_to(self):
        balls, result_image = self.bot.detectBalls(visualise=True)
        if result_image is not None:
            cv2.imwrite("test_results/detection_image.jpg", result_image)
            if len(balls) > 0:
                target = balls[0]
                print(target)
                self.bot.travelTo(target)
                np.testing.assert_allclose(self.bot.pos, target, atol=1e-1)
            else:
                print("No balls found")

    
    @unittest.skipIf("rotation_delta" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_rotation_delta(self):
        a = np.vstack(np.radians(np.arange(0, 361, 45)))
        points = np.hstack((np.cos(a), np.sin(a)))
        expected = [0, 45, 90, 135, 180, -135, -90, -45, 0]

        for (i, angle) in enumerate(expected):
            with self.subTest(f"Test rotation delta to point: {np.round(points[i], 3)})"):
                self.assertAlmostEqual(self.bot.calculateRotationDelta(points[i]), angle)
    
    @unittest.skipIf("rotation_matrix" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_rotation_matrix(self):
        point = np.sqrt(np.array([2, 2]))
        self.bot.th = 45
        # Rotates points from camera coordinates to world coordinates
        R = self.bot._getRotationMatrix()
        np.testing.assert_allclose(R @ point, np.array([2., 0.]), atol=1e-7)
    
    @unittest.skipIf("ball_detection" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_ball_detection(self):
        # Open image(s) and pass to function
        img = cv2.imread('CV/test_imgs/test_images/testing0001.jpg')
        res = self.bot.detectBalls(img)
        print(res)
        # np.testing.assert_allclose(res, np.array([
        #     [0.071, 0.91]
        # ]), atol=0.005)
    
    @unittest.skipIf("ball_detection_cam" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_ball_detection_cam(self):
        # Open image(s) and pass to function
        res = self.bot.detectBalls()
        print(res)
        # np.testing.assert_allclose(res, np.array([
        #     [0.071, 0.91]
        # ]), atol=0.005)
    
     

    @unittest.skipIf("laser" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_laser(self):

        for i in range(0, 15):
            dist = tof.read() # distance in mm
            print( i + ": " + str(dist) + "mm")
            time.sleep(0.1)   

    @unittest.skipIf("dump_balls" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_dump_balls(self):
        print("Dumping balls...")
        self.bot.dump_balls()
    
    @unittest.skipIf("collect_ball" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_collect_ball(self):
        print("Collecting ball...")
        self.bot.collect_ball()
    
    @unittest.skipIf("collect_and_dump" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_collect_and_dump(self):
        """
        Ensures collecting and dumping is done in the right order
        """
        self.bot.collect_ball()
        self.bot.dump_balls()

    @unittest.skipIf("detect_travel_collect_dump" not in ACTIVE_TESTS, "left_motor test skipped")
    def test_detect_travel_collect_dump(self):
        """
        Test everything! (except box)
        """
        res = self.bot.detectBalls()
        if len(res) > 0:
            self.bot.travelTo(res[0])
            self.bot.collect_ball()
            self.bot.dump_balls()
        else:
            print("No balls found")

if __name__ == '__main__':
    unittest.main()
    