from time import sleep
import machine

#with rpigpio library
import pigpio

IR_in = 11 #pin number signal
IR_out = 13 #pin number LED

pi.set_mode(IR_in, pigpio.INPUT)
pi.set_mode(IR_out, pigpio.OUTPUT)

IR_signal = pi.read(IR_in)
pi.write(IR_out, 1) #sets LED high




#can adjust potentiometer to adjust sensitivity
#can be used with any microcontroller (rPi or arduino)

onboard_led = machine.Pin(25, machine.Pin.OUT)
line_sensor = machine.Pin(0, machine.Pin.IN)

while True:
    if line_sensor.value() == 0:
        onboard_led.on()
    else:
        onboard_led.off()
        sleep(0.1)
