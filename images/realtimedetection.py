import cv2#Imports the OpenCV library, which is used for computer vision tasks, 
from keras.models import model_from_json #a function
#A deep learning library used for building and training neural networks
import numpy as np  

#pretrained model is loaded into the model_from_json function
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# weights are multi-dimensional arrays or matrices that determine strength bw neurons
model.load_weights("facialemotionmodel.h5")
# Load the Haar Cascade Classifier for face detection(pre trained classifier to detect faces in image)
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)
#haar_file is the path to xml file containing the pretrained haar cascade classifier
#face_cascade is object initialized with the haar cascade file


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0 #Normalize

webcam=cv2.VideoCapture(0)
# labels dictionary is used to convert the numerical class indices produced by the model's predictions into human-readable expression labels
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
while True:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
     # Detect faces in the image using the Haar Cascade Classifier and detect multiscale function returns the coordinates of the face
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try: 
        #Loop for processing each face individually
        for (p,q,r,s) in faces:
        # Extract the face region from the grayscale image
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            
            image = cv2.resize(image,(48,48))#resizing to match the i/p size req by CNN model
            img = extract_features(image)#converted to feature array
            pred = model.predict(img)# prdection is made 
            prediction_label = labels[pred.argmax()]#predection is mapped from the labels dictionary
            # print("Predicted Output:", prediction_label)
            # cv2.putText(im,prediction_label)
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))#predection label is put 

        cv2.imshow("Output",im)#displaying the processed frame and detected face and expression label 
        cv2.waitKey(27)#waiting for an keyboard interruption
    except cv2.error: #except keyword specify the type of error that is encountered and when it matches the code inside the body  of except is evaluated
        pass
 