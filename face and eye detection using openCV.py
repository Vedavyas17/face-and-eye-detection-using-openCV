#FACE AND EYE DETECTION USING OPENCV
import cv2
#cascade classifier: detects objects of different sizes in the input image
#opencv already contains many pre trained classifiers for face,eyes,liscence plate detection,smile etc.those are xmlfiles.
face_cascade=cv2.CascadeClassifier('C:/Users/vedavyas/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('C:/Users/vedavyas/Anaconda3/Lib/site-packages/cv2/data/haarcascade_eye.xml')
def detect(gray,frame):
    #syntax of detectmultiplescale is detectMultipleScale(image,rejectlevels,levelweights)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        #here scale factor=1.1,min neighbours=3 it will in range[3,5].
        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,3)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame
#to activate the webcamera
video_capture=cv2.VideoCapture(0)
while True:
    #capture frame by frame
    _,frame=video_capture.read()
    #our operations on the frame come here
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    #display the resulting frame
    cv2.imshow('video',canvas)
    #ord('q') returns the unicode point of q.cv2.waitkey(1) returns the a 32 bit integer corresponding to the pressed key
    #and &0xFF is the bit mask which sets the left 24 bits to zero,because ord() returns a valve between 0 and 255.therefore
    #once the mask is applied,it is possible to check if it is the the coresponding key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#this will make the camera to close after performing the actions[when everything done,release the video capture]
video_capture.release()
#finally it will simply destroy all the windows that we have created.if u want to destroy any specific window use cv2
cv2.destroyAllWindows()