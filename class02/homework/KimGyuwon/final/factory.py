#!/usr/bin/env python3

import os
import sys
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import numpy as np
import openvino as ov
from cv2 import cv2
from iotdemo import ColorDetector, FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):

    """_오류 검출 감지 카메라를 실행시키는 함수 _

    Args:
        q (_Queue_): _카메라 프레임 정보를 담는 변수_
    """

    # MotionDetector
    det = MotionDetector()
    det.load_preset()

    # Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model(model="./resources/openvino.xml")

    # HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    start_flag = True
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO:Cam1 detected', detected))

        # Inference OpenVINO
        input_tensor = np.expand_dims(detected, 0)

        if start_flag is True:
            ppp = ov.preprocess.PrePostProcessor(model)

            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400

            ppp.input().preprocess().resize(
                ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)

            model = ppp.build()

            compiled_model = core.compile_model(model=model, device_name="CPU")
            start_flag = False

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)

        if probs[0] > 0.0:
            print("Good Item!")
        else:
            print("Bad Item!")
            q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def thread_cam2(q):

    """_색 검출 카메라 모듈 함수_

    Args:
        q (_Queue_): _프레임을 담는 변수_
    """

    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("./resources/color.cfg", "default")

    # Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("./resources/conveyor.mp4")
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info
        q.put(('VIDEO:Cam2 detected', detected))

        # Detect color
        predict = color.detect(detected)

        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio*100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == "blue" and ratio > .5:
            q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):

    """받은 Queue를 title에 맞게 송출하는 함수"""

    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():

    """메인 함수"""

    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # Create a Queue
    que = Queue()

    # Create thread_cam1 thread_cam2 threads and start them.
    m_thread_cam1 = threading.Thread(target=thread_cam1, args=(que,))
    m_thread_cam2 = threading.Thread(target=thread_cam2, args=(que,))

    m_thread_cam1.start()
    m_thread_cam2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # get an item from the queue.
            # You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = que.get_nowait()
            except Empty:
                continue

            # HW2 show videos with titles of
            # 'Cam1 live' and 'Cam2 live' respectively.
            name, data = event
            if name.startswith("VIDEO:"):
                imshow(name[6:], data)

            # Control actuator, name == 'PUSH'
            elif name == "PUSH":
                ctrl.push_actuator(data)
                print(data)

            if name == 'DONE':
                FORCE_STOP = True

            que.task_done()

    cv2.destroyAllWindows()
    m_thread_cam1.join()
    m_thread_cam2.join()


if __name__ == '__main__':
    try:
        main()
    except ImportError:
        os._exit()
