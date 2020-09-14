import cv2
from model import MaskDetector
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = MaskDetector("model.json", "model-017.model")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(-1)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        ret, img = self.video.read()
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(grayscale_img, 1.3, 5)
        text_dict={0:'Mask ON',1:'No Mask'}
        rect_color_dict={0:(0,255,0),1:(0,0,255)}
       
        
        for (x,y,w,h) in faces:
    
            face_img = grayscale_img[y:y+w,x:x+w]
            resized_img = cv2.resize(face_img,(100,100))
            normalized_img = resized_img/255.0
            reshaped_img = np.reshape(normalized_img,(1,100,100,1))
            result=model.predict(reshaped_img)

            label=np.argmax(result,axis=1)[0]
      
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)
            cv2.putText(img, text_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2) 


        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
