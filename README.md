# ECE4191_T26

### World Coordinate System

![world coordinates](world_coordinates.jpg)

### Libraries
#### pigpio
- needs to be started, terminal command: 
```
sudo pigpiod
```
- python library that handles GPIO on a low-level
- Can produce hardware-timed PWM signals on any GPIO
- Known issue: encoder counts pause rarely and have to restart pigpio in terminal
```
sudo killall pigpiod
sudo pigpiod
```

#### ultralytics
- Loads YOLO model
- Seems quite slow so may be worth investigating other options

### Code Overview

#### [main](main.py)
High level main file to handle ALL logic relevant to the final competion

#### [camera](camera.py)
Object to handle all things computer vision including camera settings/capturing, detection algorithms and coordinate transforms

#### [robot](robot.py)
Object that implements instructions given to the robot (note that all *competition logic* should be kept in main)

#### [world](world.py)
Object that represents current state of the world. Includes tennis court dimensions, starting poses, tennnis balls etc.
