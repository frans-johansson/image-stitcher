import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    input = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.Canny(input, 50, 100)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    # Pressing <ESC> stops the program
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()