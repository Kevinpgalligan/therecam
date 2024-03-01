"""
NEXT:
Set up audio server (Common Lisp), Python should send
messages to the server containing a float (0-1) OR a 'no sound'
message. Server can adapt the sound appropriately.

Possibly useful for communicating between processes:
    https://stackoverflow.com/questions/32796877/how-to-use-usocket-to-create-a-connection-common-lisp
"""

import cv2
import numpy as np

import argparse
import time
import random
import collections

CALIBRATE_WAIT_SEC = 3
CALIBRATE_CIRCLE_RADIUS = 30
CALIBRATE_CIRCLE_THICKNESS = 5

TARGET_THICKNESS = 3

Y_COORD_QUEUE_SIZE = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-mask", action="store_true")
    args = parser.parse_args()

    cam = cv2.VideoCapture(0)

    # Colour samples from blue ball (in room w/ kinda yellow light):
    # rgb(43, 86, 126) => hsv(209, 66, 49)
    # rgb(95, 137, 159) => hsv(201, 40, 62)
    # rgb(119, 157, 174) => hsv(199, 32, 68)
    lower_bound = np.array([190, 20, 20])
    upper_bound = np.array([215, 255, 255])
    calibrate = False
    calibrate_start = 0
    q = collections.deque(maxlen=Y_COORD_QUEUE_SIZE)
    while True:
        check, frame = cam.read()
        max_y = frame.shape[0]
        frame = cv2.flip(frame, 1)

        if calibrate:
            elapsed_time = time.time() - calibrate_start
            if elapsed_time > CALIBRATE_WAIT_SEC:
                # Sample a few points from the circle, take max & min.
                hues = sample_hues(frame, yx, CALIBRATE_CIRCLE_RADIUS, 15)
                print("sampled:", hues)
                lower_bound[0] = min(hues)
                upper_bound[0] = max(hues)
                calibrate = False
            else:
                yx = (frame.shape[1]//2, frame.shape[0]//2)
                cv2.circle(frame, yx, CALIBRATE_CIRCLE_RADIUS,
                           (0, 0, 255), CALIBRATE_CIRCLE_THICKNESS)
                cv2.putText(frame, str(round(elapsed_time, 2)), yx,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # First restrict the frame to the target colour.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            frame = cv2.bitwise_and(frame, frame, mask=mask)
            # Reduces noise.
            frame = cv2.medianBlur(frame, 5)

            # Then make it greyscale and detect circle(s). Hopefully just 1.
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(grey, 50, 200)
            contours, hierarchy = cv2.findContours(edges.copy(),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print("Found", len(contours), "contours.")
            if len(contours) > 0:
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                largest = sorted_contours[0]
                M = cv2.moments(largest)
                # Traceback (most recent call last):
                # File "cam.py", line 130, in <module>
                # main()
                # File "cam.py", line 81, in main
                #     cx = int(M['m10']/M['m00'])
                # ZeroDivisionError: float division by zero
                cx = int(M['m10']/M['m00'])
                # We're interested in the y coordinate of the circle.
                cy = int(M['m01']/M['m00'])
                q.append(cy)
                # Draw a green circle to give visual feedback on how
                # the input is being interpreted.
                cv2.circle(frame, (cx,cy), 3,
                          (0, 255, 0), TARGET_THICKNESS)
            else:
                # Clear the queue of y coordinates so that we don't
                # keep producing a sound when there's no input. But
                # this should be resilient to the ball disappearing
                # momentarily.
                if q:
                    q.popleft()
        
        if q:
            avg_y = sum(q)/len(q)
            y_fraction = (max_y-avg_y)/max_y
            print("Fraction:", y_fraction)

        cv2.imshow('video', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('c'):
            calibrate = True
            calibrate_start = time.time()
    cam.release()
    cv2.destroyAllWindows()


def sample_hues(frame, yx, radius, num_points):
    x0, y0 = yx
    result = []
    while len(result) < num_points:
        dx = random.randint(-radius+1, radius-1)
        dy = random.randint(-radius+1, radius-1)
        if dx**2+dy**2<radius**2:
            x, y = x0+dx, y0+dy
            rgb = frame[y,x]
            # Have to add extra dimensions, cv2 expects a 2d image w/ 3 channels.
            hsv = cv2.cvtColor(rgb[np.newaxis, np.newaxis, :], cv2.COLOR_BGR2HSV).flatten()
            print(x,y,hsv)
            result.append(hsv[0])
    return result

if __name__ == "__main__":
    main()
