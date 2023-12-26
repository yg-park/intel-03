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
    #motion_detector = MotionDetector()  # MotionDetector 클래스가 있다고 가정
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # TODO: Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model("./resources/openvino.xml")
    
    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("./resources/conveyor.mp4")
    
    start_flag = True	
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
             break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # TODO: Motion detect
        detected = det.detect(frame)
        
	if detected is None:
             continue

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected", detected))


        # TODO: Inference OpenVINO
        input_tensor = np.expand_dims(detected, 0)
        
        if start_flag is True:
        	ppp = ov.preprocess.PrepostProcessor(model)
        	ppp.input().tensor() \
        		.set_shape(input_tensor.shape) \
        		.set_element_type(ov.Type.u8) \
        		.set_layout(ov.Layout('NHWC')) # nopa: ECE001, N400
        	ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
        	ppp.input().model().set_layout(ov.Layout('NCHW'))
        	ppp.output().tensor().set_element_type(ov.Type.f32)
        	
        	model = ppp.build()
        	compiled_model = core.compile_model(model, "CPU")
        	start_flag = False
        	
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        
        if probs[0] > 0.0:
        	print("Bad Item!")        	
      		# TODO: in queue for moving the actuator 1
        	q.put(("PUSH", 1))
        else:
        	print("Good Item!")        	

    cap.release()
    q.put(('DONE', None))
    exit()
def thread_cam2(q):
    # TODO: MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # TODO: ColorDetector
    color = ColorDetector()  # ColorDetector 클래스가 있다고 가정
    color.load_preset("./resources/color.cfg", "default")
    
    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # TODO: Detect motion
        detected = det.detect(frame)
	if detected is None:
		continue
		
        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))

        # TODO: Detect color
        predict = color.detect(detected)
        
        # TODO: Compute ratio
        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2
        if name == "blue" and ratio > .5:
        q.put(("PUSH", 2))

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
    que = Queue()

    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    thread1 = threading.Thread(target = thread_cam1, args = (que, ), daemon = True)
    thread2 = threading.Thread(target = thread_cam2, args = (que, ), daemon = True)
    
    thread1.start()
    thread2.start()
    
    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
            	event = que.get_nowait()
            except Empty:
            	continue
            	
            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            name, data = event
            if name.startswith("VIDEO:"):
            	imshow(name[6:], data)
            elif name == "PUSH":           
            # TODO: Control actuator, name == 'PUSH'
            ctrl.push_actuator(data)
            elif name == 'DONE':
            	FORCE_STOP = True

            que.task_done()
            
    thread1.join()
    thread2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
