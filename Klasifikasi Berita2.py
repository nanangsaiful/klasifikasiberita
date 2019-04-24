from keras import callbacks
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss,classification_report,f1_score,zero_one_loss
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopword = [line.rstrip('\n\r') for line in open('stopwords.txt')]
import pickle 
import os
from sklearn.metrics import f1_score
from timeit import default_timer as timer

def loadfolder(foldername):
#    label=[]
#    data=[]
#    k=0;
#    for folder in os.listdir(foldername):
#        for file in os.listdir(foldername+"/"+folder):
#                data.append(open(foldername+"/"+folder+"/"+file, "r").read())
#                label.append(k)
#        k+=1
#    df = pd.DataFrame({'data':data,'label':label})
    df= pd.read_csv("databerita.csv")
    
    return df

data=loadfolder("Dataset Berita")
print ("preposesing")
#lower
start = timer()
data['data']=data['data'].apply(lambda x: x.lower())
#mengambil huruf saja
data['data']=data['data'].apply(lambda x: re.sub('[^a-zA-z\s]','',x))
data['data']=data['data'].apply(lambda x:" ".join([stemmer.stem(y) for y in x.split(" ")] ))
data['data']=data['data'].apply(lambda x: " ".join([item for item in x.split(" ") if item not in stopword]))
kf=KFold(n_splits=5)
k=0;
rata=0 
tohammingloss=[]
hm=0
report=[]
f1=0
loss01=0
print("training")
for train, test in kf.split(data):
    train_data = np.array(data)[train]
    test_data = np.array(data)[test]
    X_train=pd.Series(np.resize(train_data[:,[0]],(train_data[:,[0]].size,)), dtype='str')
#    print("train: ",X_train)
    Y_train=train_data[:,[1]]
    Y_train=Y_train.astype(dtype="int32")
    X_test=pd.Series(np.resize(test_data[:,[0]],(test_data[:,[0]].size,)), dtype='str')
    Y_test=test_data[:,[1]]
    Y_test=Y_test.astype(dtype="int32")
    tokenizer = Tokenizer( split=' ')
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_matrix(X_train, mode='tfidf')
    X_test=tokenizer.texts_to_matrix(X_test, mode='tfidf')
#    count buat tf
    
#    print(X_train.shape,Y_train.shape)
#    print(X_test.shape,Y_test.shape)
#    print ("save tokenizer")
    with open('tokenizer250.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    model = Sequential()
#    model.add(Embedding(len(tokenizer.word_index)+1, embed_dim,input_length = X_train.shape[1]))
#    model.add(SpatialDropout1D(0.4))
#    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
#    model.add(Dense(3,activation='softmax'))
#    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
#    print(model.summary())
    
    print("build model")
    model = Sequential()
    model.add(Dense(100, activation='sigmoid', input_shape=(X_train.shape[1],)))
#    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(12,activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer="adam",metrics = ["accuracy"])
    

    batch_size = 1
    model.fit(X_train,pd.get_dummies(Y_train.ravel()).values, epochs = 10, batch_size=batch_size, verbose = 2)
    
    print("save model")
    # save model ke  JSON
    model_json = model.to_json()
    with open("model250.json", "w") as json_file:
        json_file.write(model_json)
        
    # save model
    model.save("model250.h5")
    #numpy.save("X_test",X_test)
    #numpy.save("Y_test",Y_test)
    print("Saved model to disk")
    

    y_pred = model.predict_classes(X_test)
    report.append(classification_report(Y_test, y_pred))
    f1 += f1_score(Y_test, y_pred, average='macro')
end = timer()
print("Waktu :",end - start)
print("Akurasi :", rata/5)
print("F1 Score :",f1/5)
## save the model to disk
#filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))
# 
## some time later...
# 
## load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)