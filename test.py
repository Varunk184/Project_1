#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from keras.models import load_model


# In[2]:


model=load_model('model_file.h5')


# In[11]:


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[12]:


labels_dict={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprice'}


# In[ ]:





# In[ ]:





# In[13]:



frame=cv2.imread("happyboy.jpg")
# ---------- Change image here if u want to -----------------

gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces= faceDetect.detectMultiScale(gray, 1.3, 3)
for x,y,w,h in faces:
    sub_face_img=gray[y:y+h, x:x+w]
    resized=cv2.resize(sub_face_img,(48,48))
    normalize=resized/255.0
    reshaped=np.reshape(normalize, (1, 48, 48, 1))
    result=model.predict(reshaped)
    label=np.argmax(result, axis=1)[0]
    print(label)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
    cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
cv2.imshow("Frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




