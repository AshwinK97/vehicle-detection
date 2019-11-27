import cv2

cv2Font = cv2.FONT_HERSHEY_SIMPLEX

def getArea(x, y, w, h):
    return (x+w) * (y+h)

def toString(n):
    return str(n)

# capture frames from a video
vc = cv2.VideoCapture('videos/video2.mp4')
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False


# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cascades/cars.xml')

# loop runs if new frame exists
while rval:
    # read frame from a video
    rval, frame = vc.read()
    if not rval:
        continue

    # convert to gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)[:5]
    
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(frame, toString(getArea(x, y, w, h)) + "p2", (x, y-5), cv2Font, 0.3, (0, 0, 255), 1)
    
    # Display frames in a window
    cv2.namedWindow('feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('feed', 1280, 750)
    cv2.imshow('feed', frame)
    
    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
vc.release()