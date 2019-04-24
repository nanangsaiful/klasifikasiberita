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
for skenariostem in ["nostem"]:#,'nostem'
    for skenariocount in ['tfidf','count']:#'count',
        for skenarioopti in ["lbfgs","adam","sgd"]:
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
            rata=0
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
                
    #            print(X_train.shape,Y_train.shape)
    #            print(X_test.shape,Y_test.shape)
            #    print ("save tokenizer")
                with open('tokenizer250'+skenariostem+str(k)+'.pickle', 'wb') as handle:
                    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=75, activation="logistic",
                                    solver=skenarioopti, random_state=1, verbose = False,
                                    learning_rate_init=.1)
                
                mlp.fit(X_train, Y_train.ravel())
            #    save the model to disk
              
                filename = 'final_model'+skenariocount+skenariostem+skenarioopti+str(k)+'.sav'
                pickle.dump(mlp, open(filename, 'wb+'))
#                print("Training set score: %f" % mlp.score(X_train, Y_train.ravel()))
                rata += mlp.score(X_test, Y_test)
#                print("Test set score: %f" % mlp.score(X_test, Y_test.ravel()))
                y_pred = mlp.predict(X_test)
                report.append(classification_report(Y_test, y_pred))
                f1ma.append(f1_score(Y_test, y_pred, average='macro'))
                f1mi.append(f1_score(Y_test, y_pred, average='micro'))
                k+=1
            end = timer()
            print("skenario "+skenariocount +skenariostem+skenarioopti)
            print("Waktu :",end - start)
            print("Akurasi :", rata/5)
            print("F1 Score Macro :",np.mean(f1ma))
            print("F1 Score Micro :",np.mean(f1mi))
        
    ## some time later...
    # 
    ## load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    #print(result)