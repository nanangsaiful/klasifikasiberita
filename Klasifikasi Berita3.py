
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,f1_score,accuracy_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopword = [line.rstrip('\n\r') for line in open('stopwords.txt')]
import pickle 
import os
from timeit import default_timer as timer

def loadfolder(foldername):
    label=[]
    data=[]
    k=0;
    for folder in os.listdir(foldername):
        for file in os.listdir(foldername+"/"+folder):
                data.append(open(foldername+"/"+folder+"/"+file, "r").read())
                label.append(k)
        k+=1
    df = pd.DataFrame({'data':data,'label':label})
    return df
for skenariostem in ['nostem']:#,'nostem'
    for skenariocount in ['tfidf']:#'count',
        data=loadfolder("Dataset Berita")
        print ("preposesing")
        #lower
        start = timer()
        data['data']=data['data'].apply(lambda x: x.lower())
        #mengambil huruf saja
        data['data']=data['data'].apply(lambda x: re.sub('[^a-z\s]',' ',x))
        if skenariostem=="stem":
            data['data']=data['data'].apply(lambda x:[stemmer.stem(y) for y in x.split(" ")] )
        else:
            data['data']=data['data'].apply(lambda x: x.split(" ") )
        data['data']=data['data'].apply(lambda x:[item for item in x if ((len(item)>2) & (item not in stopword)) ])

        kf=KFold(n_splits=5)
        k=0;
        np.random.seed(25)
        indextukar=np.random.permutation(len(data))
        np.save('hasilindextukar',[indextukar,kf])
        data=data.iloc[indextukar]
        report=[]
        rata=[]
        f1mi=[]
        f1ma=[]
        for train, test in kf.split(data):
            train_data = np.array(data)[train]
            test_data = np.array(data)[test]
            X_train=pd.Series(np.resize(train_data[:,[0]],(train_data[:,[0]].size,)))
        #    print("train: ",X_train)
            Y_train=train_data[:,[1]]
            Y_train=Y_train.astype(dtype="int32")
            X_test=pd.Series(np.resize(test_data[:,[0]],(test_data[:,[0]].size,)))
            Y_test=test_data[:,[1]]
            Y_test=Y_test.astype(dtype="int32")
            tokenizer = Tokenizer( split=' ')
            tokenizer.fit_on_texts(X_train)
            
            X_train = tokenizer.texts_to_matrix(X_train, mode=skenariocount)
            X_test=tokenizer.texts_to_matrix(X_test, mode= skenariocount )
        #    count buat tf
            Y_train=pd.get_dummies(Y_train.ravel()).values
            Y_test=pd.get_dummies(Y_test.ravel()).values        
        #    print ("save tokenizer")
            with open('tokenizer250'+skenariostem+str(k)+'.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            model = Sequential()
            model.add(Dense(100, activation='sigmoid', 
                            input_shape=(X_train.shape[1],)))
            #model.add(Dense(50, activation='sigmoid'))binary_crossentropy
            model.add(Dense(12, activation='sigmoid'))
            sgd = optimizers.SGD(lr=0.1 , momentum=0.9, nesterov=True)
            model.compile(loss = 'categorical_crossentropy', 
                          optimizer=sgd,metrics = ["accuracy"])
            batch_size = 200
            ckpt=ModelCheckpoint("model.h5", monitor="val_acc",
                             verbose=1,save_best_only=True)
            es=EarlyStopping(monitor="loss",verbose=1,patience=5)
            model.fit(X_train, Y_train, epochs = 50, 
                             batch_size=batch_size,verbose = 2,
                             validation_data=(X_test,Y_test),
                             callbacks=[ckpt,es])            
            print(X_train.shape,Y_train.shape)
            print(X_test.shape,Y_test.shape)
            

                
            # save model
            model=load_model("model.h5")
            
            labelhasil=model.predict_classes(X_test)
            report.append(classification_report(Y_test.argmax(axis=1),
                                     labelhasil.astype("int64")))
            rata.append(accuracy_score(Y_test.argmax(axis=1),
                                       labelhasil.astype("int64")))
            f1mi.append(f1_score(Y_test.argmax(axis=1),labelhasil.astype("int64")
                      ,average="micro"))
            f1ma.append(f1_score(Y_test.argmax(axis=1),
                      labelhasil.astype("int64"),average="macro"))
            k+=1
        end = timer()
        print("skenario "+skenariocount +skenariostem)
        print("Waktu :",end - start)
        print("Akurasi :", np.mean(rata))
        print("F1 Score Macro :",np.mean(f1ma))
        print("F1 Score Micro :",np.mean(f1mi))
        
