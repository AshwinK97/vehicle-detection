import cv2

# store each detected vehicles info in 2D list
vehicles = [] # [xPos, yPos, width, height, area, velocity, age]

def getArea(x, y, w, h):
    return (x+w) * (y+h)

# get velocity assuming video is in 30fps
def getVelocity(p1, p2):
    return float(p2[0] - p1[0]) / (p2[1] - p2[1])/(float(1/30))

def toString(n):
    return str(n)

def addVehicle(x, y, w, h, a):
    # check if x, y, w, h are within 5-10 px of existing vehicle
    for v in vehicles:
        if abs(x-v[0]) <= 5 and abs(y-v[1]):
            v = [x, y, w, h, getArea(x, y, w, h), getVelocity([x, y], [v[0], v[1]]), 0] # updated vehicle
            return
    # if similar vehicle is not found, add new one
    vehicles.append([x, y, w, h, a, 0, 0])

def vehicleAge():
    for v in vehicles:
        if v[6] >= 30:
            vehicles.remove(v)
        else:
            v[6] += 1


cv2Font = cv2.FONT_HERSHEY_SIMPLEX
car_cascade = cv2.CascadeClassifier('cars.xml')
vc = cv2.VideoCapture('video2.mp4')
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
    
    # draw rectangles
    for (x,y,w,h) in cars:
        addVehicle(x, y, w, h, getArea(x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(frame, toString(getArea(x, y, w, h)) + "p2", (x, y-5), cv2Font, 0.3, (0, 0, 255), 1)
    
    print(vehicles)
    print('\n')
    print('\n')
    vehicleAge()
    
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