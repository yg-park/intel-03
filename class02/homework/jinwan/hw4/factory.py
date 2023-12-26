#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino.inference_engine import IECore

from iotdemo import FactoryController

FORCE_STOP = False


def thread_cam1(q):
    # TODO: MotionDetector
    motion_detector = MotionDetector()  # MotionDetector 클래스가 있다고 가정ddd

    # TODO: Load and initialize OpenVINO
    ie = IECore()
    net = ie.read_network(model='path/to/model.xml', weights='path/to/model.bin')
    exec_net = ie.load_network(network=net, device_name='CPU', num_requests=1)


    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('resources/conveyor.mp4')


    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(('VIDEO:Cam1 live', frame_info))


        # TODO: Motion detect
        motion_detected = motion_detector.detect(frame)
	if motion_detected:
    		q.put(('VIDEO:Cam1 detected', detected_info))

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO:Cam1 detected', detected_info))

        # abnormal detect
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reshaped = detected[:, :, [2, 1, 0]]
        np_data = np.moveaxis(reshaped, -1, 0)
        preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO: Inference OpenVINO
        inference_result = exec_net.infer(inputs={'input': batch_tensor})


        # TODO: Calculate ratios
        x_ratio = inference_result['x_ratio']
	circle_ratio = inference_result['circle_ratio']
	
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1
        q.put(('PUSH', 1))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    motion_detector = MotionDetector()  # MotionDetector 클래스가 있다고 가정


    # TODO: ColorDetector
    color_detector = ColorDetector()  # ColorDetector 클래스가 있다고 가정


    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('resources/conveyor.mp4')


    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:Cam2 live', frame_info))


        # TODO: Detect motion
        motion_detected = motion_detector.detect(frame)
	if motion_detected:
    		q.put(('VIDEO:Cam2 detected', detected_info))

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO:Cam2 detected', detected_info))


        # TODO: Detect color
        color_detected = color_detector.detect_color(frame)
	if color_detected:
    		q.put(('VIDEO:Cam2 color detected', color_detected_info))


        # TODO: Compute ratio
        ratio = color_detector.compute_ratio(frame)

        print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2
        q.put(('PUSH', 2))


    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # TODO: HW2 Create a Queue

    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data

            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.

            # TODO: Control actuator, name == 'PUSH'

            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
