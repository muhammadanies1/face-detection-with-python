import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640) #code number 3 its for width
cam.set(4, 480) #code number 4 its for height
faceDirectory = 'images/ORL'
faceDetector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# faceID = input('Insert your Id Number: ')
# faceName = input('Insert your name: ')
print ('Please waiting...')
dataSet = 45

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1) #vertical flip
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(grey, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w, y+h), (0,225,225), 2)
        # fileName = 'faces_'+str(faceID)+'_'+str(faceName)+'_'+str(dataSet)+'.jpg'
        fileName = str(dataSet)+'.jpg'
        cv2.imwrite(faceDirectory+'/'+fileName, frame)
        dataSet += 1
        roiGrey = grey[y:y+h, x:x+w]
        roiColor = frame[y:y+h, x:x+w]
        eyes = eyeDetector.detectMultiScale(roiGrey)
        for (xe, ye, he, we) in eyes:
            cv2.rectangle(roiColor, (xe, ye), (we+we, ye+he), (0,225,225), 1)
    
    # cv2.imshow("My Camera", frame)
    # cv2.imshow("My Camera Gray", grey)
    exit = cv2.waitKey(1) & 0xFF
    if exit == 27 or exit == ord('q'):
        break
    elif dataSet>60:
        break
print ('Data recorded!')

cam.release()
cv2.destroyAllWindows()