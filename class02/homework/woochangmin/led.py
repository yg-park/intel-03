"""
Smart Factory controller APIs
"""
from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

while True:
    user_input = input('please enter input: ')
    if user_input == '1':
        ctrl.system_start()
    if user_input == '2':
        ctrl.system_stop()
    if user_input == '3':
        ctrl.red=ctrl.red
    if user_input == '4':
        ctrl.orange=ctrl.orange
    if user_input == '5':
        ctrl.green=ctrl.green
    if user_input == '6':
        ctrl.conveyor=ctrl.conveyor
    if user_input == '7':
        ctrl.push_actuator(1)
    if user_input == '8':
        ctrl.push_actuator(2)
    if user_input == 'x':
        ctrl.system_stop()
        break

ctrl.close()
