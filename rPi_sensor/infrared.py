from time import sleep
import machine

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