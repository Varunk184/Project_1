#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os


# In[2]:


# DATASET LOCATION - (FER2013)
train_data_dir='C://Users//varun//OneDrive//Desktop//fer2013//train'
test_data_dir='C://Users//varun//OneDrive//Desktop//fer2013//test'


# In[3]:


train_datagen=ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                shear_range=0.3,
                zoom_range=0.3,
                horizontal_flip=True,
                fill_mode='nearest')


# In[4]:


validation_datagen=ImageDataGenerator(rescale=1./255)


# In[5]:


train_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    color_mode='grayscale',
                    target_size=(48,48),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=True)


# In[6]:


validation_datagen=validation_datagen.flow_from_directory(
                    test_data_dir,
                    color_mode='grayscale',
                    target_size=(48,48),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=True
)


# In[7]:


class_lables=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']


# In[8]:


img,label=train_generator.__next__()


# In[23]:


model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[24]:


print(model.summary())


# In[25]:


# DATASET LOCATION - (FER2013)

train_path='C://Users//varun//OneDrive//Desktop//fer2013//train'
test_path='C://Users//varun//OneDrive//Desktop//fer2013//test'


# In[26]:


num_train_imgs=0
for root,dirs,files in os.walk(train_path):
    num_train_imgs+=len(files)
    
num_test_imgs=0
for root,dirs,files in os.walk(test_path):
    num_test_imgs+=len(files)


# In[27]:


print(num_train_imgs,num_test_imgs)


# Change epochs to 100 if u want to increase the accuracy of the model

# In[28]:


# Change Epochs here 


epochs=30
history=model.fit(train_generator,
                 steps_per_epoch=num_train_imgs//32,
                 epochs=epochs,
                 validation_data=validation_datagen,
                 validation_steps=num_test_imgs//32)
model.save('model_file.h5')


# In[ ]:





# In[ ]:




