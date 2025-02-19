import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()
height, width = frame.shape[:2]

border = 20
x1, y1 = border, border
x2, y2 = width - border, height - border

frame_border_offset = 5

frame_border_thickness = 2

canvas = np.zeros((height, width, 4), dtype=np.uint8)

prev_cX = None
prev_cY = None

erase_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()

    cv2.rectangle(frame_copy, (x1 - frame_border_offset, y1 - frame_border_offset),
                  (x2 + frame_border_offset, y2 + frame_border_offset), (0, 0, 255), frame_border_thickness)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([179, 255, 255])


    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])


    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)


    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_blue:
        erase_mode = True
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        prev_cX = None
        prev_cY = None

    if not erase_mode:
        for contour in contours_red:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if x1 <= cX <= x2 and y1 <= cY <= y2:
                    if prev_cX is not None and prev_cY is not None and \
                       x1 <= prev_cX <= x2 and y1 <= prev_cY <= y2 and \
                       (abs(cX - prev_cX) > 1 or abs(cY - prev_cY) > 1):
                        cv2.line(canvas, (prev_cX, prev_cY), (cX, cY), (0, 255, 0, 255), 2)

                    prev_cX = cX
                    prev_cY = cY
    elif not contours_blue:
        erase_mode = False

    if not contours_red and not erase_mode:
        prev_cX = None
        prev_cY = None

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGR)

    mask_lines = cv2.cvtColor(canvas, cv2.COLOR_RGBA2GRAY)
    _, mask_lines = cv2.threshold(mask_lines, 1, 255, cv2.THRESH_BINARY)
    mask_lines = cv2.cvtColor(mask_lines, cv2.COLOR_GRAY2BGR)

    mask_lines = cv2.bitwise_not(mask_lines)

    black_background = np.zeros_like(frame)
    result = cv2.bitwise_and(mask_lines, black_background)
    result = cv2.add(result, canvas_bgr)

    cv2.rectangle(result, (x1 - frame_border_offset, y1 - frame_border_offset),
                  (x2 + frame_border_offset, y2 + frame_border_offset), (0, 0, 255), frame_border_thickness)

    cv2.imshow("Result", result)
    cv2.imshow("o", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()