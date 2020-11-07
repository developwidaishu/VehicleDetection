import cv2

#capturing video frames
cap = cv2.VideoCapture('sample1.mp4')

#Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')

#loop runs if capturing has initiated
while True:
    #reads frames from a video
    ret,frames = cap.read()

    #convert to grayscale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    #detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    #drawing rectangles on each car
    for (x, y, w, h) in cars:
        cv2.rectangle(frames,(x, y),(x+w, y+h),(0,0,255), 2)
    
    #Display frames in a window
    cv2.imshow('video2', frames)

    #Wait for esc key to stop
    if cv2.waitKey(33) == 27:
        break

#De-allocate any associated memory usage
cv2.destroyAllWindows()
