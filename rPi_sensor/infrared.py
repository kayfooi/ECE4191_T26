from time import sleep
# import machine

#with rpigpio library
import pigpio

pi = pigpio.pi()
IR__ball_detect_in = 5 # GPIO Number for IR Sensor (S pin)
pi.set_mode(IR__ball_detect_in, pigpio.INPUT)

#testing IR
while True:
    IR_signal = pi.read(IR__ball_detect_in)

    if IR_signal == 0:
        print("Detected")
    else:
        print("Not Detected")
    
    sleep(0.2)



def IR_init(IR_in, IR_out): 
    for pins_in in IR_in:
        pigpio.pi.set_mode(pins_in, pigpio.INPUT)


    for pins_out in IR_out:
        pigpio.pi.set_mode(pins_out, pigpio.OUTPUT)

def IR_read(IR_in, IR_out):
    for i in IR_in:
        IR_signal.append(pigpio.pi.read(i))
    for i in IR_out:
        
        if IR_signal[IR_out.index(i)] == 0:
            pigpio.pi.write(i, 1)
        else:
            pigpio.pi.write(i, 0)
        


#can adjust potentiometer to adjust sensitivity
#can be used with any microcontroller (rPi or arduino)

# onboard_led = machine.Pin(25, machine.Pin.OUT)
# line_sensor = machine.Pin(0, machine.Pin.IN)

# while True:
#     if line_sensor.value() == 0:
#         onboard_led.on()
#     else:
#         onboard_led.off()
#         sleep(0.1)
