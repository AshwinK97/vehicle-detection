import cv2

cv2Font = cv2.FONT_HERSHEY_SIMPLEX
windowName = 'feed'
vehicles = [] # 2D list, each item => [xPos, yPos, width, height, area, speed, age]


def getArea(x, y, w, h):
    return (x+w) * (y+h)

# get velocity assuming video is in 30fps
def getSpeed(a1, a2):
    # return (a2 - a1)/(1/30)
    return (a2 - a1) / (30)

def toString(n):
    return str(n)

# add vehicles to list or update existing vehicle with new position and speed
def addVehicle(x, y, w, h, a):
    # check if x, y, w, h are within 20 - 50 px of existing vehicle
    for v in vehicles:
        if abs(x-v[0]) <= 25 and abs(y-v[1]) <= 25:
            area = getArea(x, y, w, h)
            speed = getSpeed(area, v[4])
            vehicles[vehicles.index(v)] = [x, y, w, h, area, speed, 0] # update vehicle
            return
    vehicles.append([x, y, w, h, a, 0, 0]) # add new vehicle

# increase vehicle age until 30 (1s), then remove
def vehicleAge():
    for v in vehicles:
        if v[6] >= 30:
            vehicles.remove(v)
        else:
            v[6] += 1

def printVehicles():
    for i, v in enumerate(vehicles):
        print("%d) x: %d  y: %d  velocity: %.2f px/s" % (i+1, v[0], v[1], float(v[5])))
    print("")


car_cascade = cv2.CascadeClassifier('cars.xml')
vc = cv2.VideoCapture('video7.mp4')
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    if not rval:
        continue

    # Detects cars of different sizes in grayscale image
    cars = car_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=4)[:5]
    
    # draw rectangles, add vehicles to list, remove if they are no longer relevant
    for (x,y,w,h) in cars:
        addVehicle(x, y, w, h, getArea(x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(frame, toString(getArea(x, y, w, h)) + "p2", (x, y-5), cv2Font, 0.3, (0, 0, 255), 1)
    vehicleAge()

    # show tracked vehicle data
    print("Tracking %d vehicles, currently detected: %d" % (len(vehicles), len(cars)))
    printVehicles()
    
    # Display frames in a window
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 1280, 750)
    cv2.imshow(windowName, frame)
    
    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
vc.release()