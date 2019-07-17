#!/usr/bin/env python
# coding: utf-8

# In[1]:


import speech_recognition as sr


# In[2]:


r = sr.Recognizer()


# In[3]:


with sr.Microphone() as source:
    print('Speak Anything :')
    audio=r.listen(source,phrase_time_limit=5);
    
    try:
        text=r.recognize_google(audio);
        print('You spoke : {0}'.format(text))
    except:
        print('Could not decipher sound ')


# In[4]:


import pyaudio
import wave

CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "audio/output10.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

no_stop_words = remove_stop_words(text)


# In[ ]:


# def get_stemmed_text(corpus):
#     from nltk.stem.porter import PorterStemmer
#     stemmer = PorterStemmer()
#     return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

# stemmed_reviews = get_stemmed_text(reviews_train_clean)


# USE CASES:
# 
# What are the most reviewed Amazon products?
# What are the initial and current number of customer reviews for each product?
# How do the reviews in the first 90 days after a product launch compare to the price of the product?
# How do the reviews in the first 90 days after a product launch compare to the days available for sale?
# Map the keywords in the review text against the review ratings to help train sentiment models.

# # Now we will build the model for Sentiment analysis and later test it on the text from voice .

# In[ ]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D,Conv2D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix


# In[ ]:



from keras import regularizers


# In[ ]:


import os


# In[ ]:


mylist=os.listdir('RawData/')


# In[ ]:


len(mylist)


# # Filename identifiers 
# 
# Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
# Vocal channel (01 = speech, 02 = song).
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

# In[ ]:


print(mylist[400])


# In[ ]:


#emotion 
print(mylist[400][6:8])


# In[ ]:


#03 Audio only 01- Speech 01 neutral 01 emotional intensity 01 = "Kids are talking by the door" 01 = 1st repetition
data, sampling_rate = librosa.load('RawData/03-01-01-01-01-01-07.wav')


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import os
import pandas as pd
import librosa
import glob 

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)


# In[ ]:


feeling_list=[]
for item in mylist:
    if item[6:-16]=='02' and int(item[18:-4])%2==0:
        feeling_list.append('female_calm')
    elif item[6:-16]=='02' and int(item[18:-4])%2==1:
        feeling_list.append('male_calm')
    elif item[6:-16]=='03' and int(item[18:-4])%2==0:
        feeling_list.append('female_happy')
    elif item[6:-16]=='03' and int(item[18:-4])%2==1:
        feeling_list.append('male_happy')
    elif item[6:-16]=='04' and int(item[18:-4])%2==0:
        feeling_list.append('female_sad')
    elif item[6:-16]=='04' and int(item[18:-4])%2==1:
        feeling_list.append('male_sad')
    elif item[6:-16]=='05' and int(item[18:-4])%2==0:
        feeling_list.append('female_angry')
    elif item[6:-16]=='05' and int(item[18:-4])%2==1:
        feeling_list.append('male_angry')
    elif item[6:-16]=='06' and int(item[18:-4])%2==0:
        feeling_list.append('female_fearful')
    elif item[6:-16]=='06' and int(item[18:-4])%2==1:
        feeling_list.append('male_fearful')
    elif item[:1]=='a':
        feeling_list.append('male_angry')
    elif item[:1]=='f':
        feeling_list.append('male_fearful')
    elif item[:1]=='h':
        feeling_list.append('male_happy')
    #elif item[:1]=='n':
        #feeling_list.append('neutral')
    elif item[:2]=='sa':
        feeling_list.append('male_sad')


# In[ ]:


labels = pd.DataFrame(feeling_list)


# In[ ]:


labels[:10]


# In[ ]:



df = pd.DataFrame(columns=['feature'])
bookmark=0
for index,y in enumerate(mylist):
    if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08':# and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d':
        X, sample_rate = librosa.load('RawData/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13),
                        axis=0)
        feature = mfccs
        #[float(i) for i in feature]
        #feature1=feature[:135]
        df.loc[bookmark] = [feature]
        bookmark=bookmark+1


# In[ ]:


df.head()


# In[ ]:


df3 = pd.DataFrame(df['feature'].values.tolist())


# In[ ]:



newdf = pd.concat([df3,labels], axis=1)


# In[ ]:


rnewdf = newdf.rename(index=str, columns={"0": "label"})


# In[ ]:


rnewdf.head()


# In[ ]:


from sklearn.utils import shuffle
rnewdf = shuffle(newdf)
rnewdf[:10]


# In[ ]:


rnewdf=rnewdf.fillna(0)


# In[ ]:


#train test split 
newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]


# In[ ]:


trainfeatures = train.iloc[:, :-1]

trainlabel = train.iloc[:, -1:]

testfeatures = test.iloc[:, :-1]

testlabel = test.iloc[:, -1:]


# In[ ]:


from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))


# In[ ]:


X_test.shape


# In[ ]:


x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)

# X_train.reshape(795,216,1)
# X_test.reshape(165,216,1)


# In[ ]:


model = Sequential()
model.add(Conv1D(256, 5,padding='same',input_shape=(216,1))) 
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)


# In[ ]:



model.summary()


# In[ ]:



model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])


# In[ ]:


cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))


# In[ ]:


plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # Save the Weights and the model 
# 

# In[ ]:


model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[ ]:


import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:


# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
# loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# # Prediction Validation 

# In[ ]:



preds = loaded_model.predict(x_testcnn, 
                         batch_size=32, 
                         verbose=1)


# In[ ]:


preds1=preds.argmax(axis=1)
abc = preds1.astype(int).flatten()


# In[ ]:


predictions = (lb.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues': predictions})
preddf


# In[ ]:


actual=y_test.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))
actualdf = pd.DataFrame({'actualvalues': actualvalues})
actualdf[:10]


# In[ ]:


finaldf = actualdf.join(preddf)
finaldf


# # After Model training and testing use it against a random audio file 

# In[ ]:



data, sampling_rate = librosa.load('audio\output10.wav')


# In[ ]:



get_ipython().run_line_magic('pylab', 'inline')
import os
import pandas as pd
import librosa
import glob 

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)


# In[ ]:


X, sample_rate = librosa.load('audio\output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive


# In[ ]:



livedf2= pd.DataFrame(data=livedf2)


# In[ ]:


livedf2


# In[ ]:


livedf2=livedf2.T


# In[ ]:


livedf2


# In[ ]:



twodim= np.expand_dims(livedf2, axis=2)


# In[ ]:


livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)


# In[ ]:


livepreds1=livepreds.argmax(axis=1)


# In[ ]:


liveabc = livepreds1.astype(int).flatten()


# In[ ]:


livepredictions = (lb.inverse_transform((liveabc)))
livepredictions


# In[ ]:


liveabc


# In[ ]:




