import cv2

cam = cv2.VideoCapture(0)
# cam.set(3, 640) #code number 3 its for width
# cam.set(4, 480) #code number 4 its for height
faceDetector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
#function for detect face call haar cascade xml
eyeDetector = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1) #vertical flip
    
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(grey, 1.3, 5) 
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w, y+h), (0,225,225), 2)
    #function make rectangle color yellow
    cv2.imshow("My Camera", frame)
    # cv2.imshow("My Camera Gray", grey)
    exit = cv2.waitKey(1) & 0xFF
    if exit == 27 or exit == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()