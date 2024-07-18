import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  
import warnings
from scipy import stats
import matplotlib.pyplot as plt 

warnings.filterwarnings("ignore")

from tensorflow import keras
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense as cnn, Input, Dropout as rnn, Convolution1D, MaxPool1D as lstm, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

def applyDL(fileName) :
    try :
        print('Apply LSTM RNN based CNN')
        
        df_train = pd.read_csv(fileName, header=None)
        idx = round(df_train.shape[0]/len(df_train))-1
        
        Y = np.array(df_train[idx].values).astype(np.int8)
        X = np.array(df_train[list(range(idx))].values)[..., np.newaxis]
        X = X.reshape((X.shape[0], X.shape[1]))
        
        nclass = len(np.unique(Y))
        inp = Input(shape=(idx, 1))
        layers = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
        layers = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(layers)
        layers = lstm(pool_size=2)(layers)
        layers = rnn(rate=0.1)(layers)
        layers = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layers)
        layers = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layers)
        layers = lstm(pool_size=2)(layers)
        layers = rnn(rate=0.1)(layers)
        layers = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layers)
        layers = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layers)
        layers = lstm(pool_size=2)(layers)
        layers = rnn(rate=0.1)(layers)
        layers = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(layers)
        layers = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(layers)
        layers = GlobalMaxPool1D()(layers)
        layers = rnn(rate=0.2)(layers)
        
        dense_1 = cnn(64, activation=activations.relu, name="dense_1")(layers)
        dense_1 = cnn(64, activation=activations.relu, name="dense_2")(dense_1)
        dense_1 = cnn(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)
        
        model = models.Model(inputs=inp, outputs=dense_1)
        opt = optimizers.Adam(0.001)
        
        model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
        model.summary()
        file_path = "mitbih_model.h5"
        train = 0
        
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
        callbacks_list = [checkpoint, early, redonplat]  # early
        
        model.fit(X, Y, epochs=1, verbose=1, callbacks=callbacks_list, validation_split=0.1)
        
        Y_test = Y
        y = model.predict(X)
        pred_test = np.argmax(y, axis=-1)
        
        f1 = f1_score(Y_test, pred_test, average="macro")
        print("Test f1 score : %s "% f1)
        pre = precision_score(Y_test, pred_test, average="macro")
        print("Test Precision score : %s "% pre)
        acc = accuracy_score(Y_test, pred_test)
        print("Test accuracy score : %s "% acc)
        rec = recall_score(Y_test, pred_test, average="macro")
        print("Test Recall score : %s "% rec)
        conf_matrix = confusion_matrix(Y_test, pred_test);
        print('Confusion matrix')
        print(conf_matrix)
        
        y_data = stats.norm.pdf(Y_test, 0, 1)
        plt.plot(Y_test, 1-y_data);
        plt.xlabel('TP')
        plt.ylabel('TN')
        plt.title('ROC')
        plt.show()
    except :
        x = 0
    
    