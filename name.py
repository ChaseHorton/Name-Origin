# -*- coding: utf-8 -*-
"""
RNN for learning last names
Created on Thu Aug  1 17:31:36 2019

@author: Chase
"""
import pandas as pd
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
tensorBoard = TensorBoard(log_dir = 'data/{}'.format(time()))
from keras.models import load_model

names = {}
raw_text = ''
simple_array =[[]]
i = 0
holder = ''

#importing names
for name in glob.glob('names/*'):
    f=open(name, 'r',encoding='utf-8')
    name = name.replace('names\\','')
    name = name.replace('.txt','')
    names.update({name : f.read()})
#placing formatting names into array
for key in names:
    for item in names[key]:
        raw_text = raw_text + item
        if item == '\n':
            simple_array.append([key, holder.lower()])
            holder = ''
        else:
            holder = holder + item
            
#making names into dataframe
simple_array = simple_array[1:][:]
df = pd.DataFrame(simple_array, columns = ['Language', 'Name'])

#making characters into integers
chars = sorted(list(set(raw_text.lower())))
int_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_int = dict((c, i) for i, c in enumerate(chars))
second_array = [[]]
third_array = []
for i in df.iloc[:,1]:
    for c in i:
        third_array.append(char_to_int[c])
    second_array.append(third_array)
    third_array = []
second_array = second_array[1:][:]
n = df.columns[1]
df.drop(n, axis = 1, inplace = True)
df[n] = second_array

#encoding names of languages
le = LabelEncoder()
x_train = df.iloc[:,1]
y_train = df.iloc[:,0].to_numpy()
y_train = le.fit_transform(y_train)
y_train = y_train.reshape(-1, 1)
x_train = pd.Series.tolist(x_train)
onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train).toarray()

#padding and reshaping 
X = np.array(x_train)
X_NEW = np.zeros((len(X),20,1))
X_NEW.shape
bigct = 0
ct = 0
for i in X:
    for c in i:
        X_NEW[bigct][ct][0]=c
        ct+=1
    ct=0
    bigct+=1
X= X_NEW
X = X/len(chars)

#finally making model
model = Sequential()
model.add(CuDNNLSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X,y_train, epochs = 50, batch_size =10, callbacks= [tensorBoard])
model.save('Model.h5')


from keras.models import load_model
model = load_model('Model.h5')
def predict(input_line, number = 3):
    FINAL_CATEGORIES = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French','German','Greek','Irish','Italian','Japanese','Korean','Polish','Portuguese','Russian','Scottish','Spanish','Vietnamese']
    x = []
    out = []
    nums = []
    index = 0
    sumA = 0
    for i in range(20):
        if i < len(input_line):
            x.append(char_to_int[input_line[i].lower()])
        else :
            x.append(0)
    x = np.reshape(x, (1, len(x), 1))
    x = x / len(chars)
    prediction = model.predict(x, verbose=0)
    prediction = prediction.tolist()
    maximum=prediction[0][0]
    for i in range(number):
        for y,value in enumerate(prediction[0]):
            if value>maximum:
                maximum=value
                index = y
        nums.append(maximum)
        out.append(FINAL_CATEGORIES[index])
        FINAL_CATEGORIES.remove(FINAL_CATEGORIES[index])
        prediction[0].remove(prediction[0][index])
        index = 0
        maximum = prediction[0][0]
    for i in range(len(nums)):
        sumA = sumA + nums[i]
        confidence = nums[0]/sumA
    print ('Confidence: ' + str(confidence))
    print(out)

while 1:
    name_in =input("Enter a name: ")
    num_of_guesses = int(input("How many guesses should I make? "))
    predict(name_in,num_of_guesses)
    if input("would you like to go again (y/n)? ") == 'n':
        break