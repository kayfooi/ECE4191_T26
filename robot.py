import numpy as np

class DiffDriveRobot:
    # Based off code provided in ECE4191 Github file called Robot_navigation_and_control.ipynb
    def __init__(self,inertia=5, dt=0.1, drag=0.2, wheel_radius=0.05, wheel_sep=0.15):
        
        self.x = 0.0 # y-position
        self.y = 0.0 # y-position 
        self.th = 0.0 # orientation
        
        self.wl = 0.0 #rotational velocity left wheel
        self.wr = 0.0 #rotational velocity right wheel
        
        self.I = inertia
        self.d = drag
        self.dt = dt
        
        self.r = wheel_radius
        self.l = wheel_sep

    # Here, we simulate the real system and measurement
    def motor_simulator(self,w,duty_cycle):
        
        torque = self.I*duty_cycle
        
        if (w > 0):
            w = min(w + self.dt*(torque - self.d*w),3)
        elif (w < 0):
            w = max(w + self.dt*(torque - self.d*w),-3)
        else:
            w = w + self.dt*(torque)
        
        return w

    # Find the motor encoder measurement to determine how fast the wheel is turining
    def motorEncoder(self):
        pass
    
    # Velocity motion model
    def baseVelocity(self,wl,wr):
        v = (wl*self.r + wr*self.r)/2.0 #linear velocity
        w = (wl*self.r - wr*self.r)/self.l #angular velocity
        
        return v, w
    
    # Kinematic motion model
    def pose_update(self,duty_cycle_l,duty_cycle_r):
        
        # self.wl = self.motor_simulator(self.wl,duty_cycle_l)
        # self.wr = self.motor_simulator(self.wr,duty_cycle_r)
        
        v, w = self.baseVelocity(self.wl,self.wr)
        
        self.x = self.x + self.dt*v*np.cos(self.th) # x = x + dt*v*cos(th)
        self.y = self.y + self.dt*v*np.sin(self.th) # y = y + dt*v*sin(th)
        self.th = self.th + w*self.dt # th = th + w*dt
        
        return self.x, self.y, self.th