# read and split the data
import h5py
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# read DNA sequences data
with h5py.File('seq_data_8000_POST.h5', 'r') as h5f:
    data = h5f['seq_data_8000_POST'][:]
    h5f.close()

# read in the label data
with h5py.File('label_data.h5', 'r') as hf:
    label = hf['label_data'][:]
    hf.close()

# split dataset into training, validation and testing sets - 70/15/15
# first split 70% into training and 30% into testing
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
# then split 30% of testing set into 15% testing and 15% validation
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=28)


layers=5
filters=32
plots_path='5CNN_LSTM_FOR/'
np.set_printoptions(precision=4)
df=pd.DataFrame(columns=['val_loss', 'val_mae'])

for window in range(10, 550, 50):
    print("\nChecking "+str(layers)+" Conv_2LSTM layers with batch_size=256, f="+str(filters)+" and w="+str(window)+":\n--------------")
    model=Sequential()
    model.add(Conv1D(filters=filters, kernel_size=window, activation='relu', input_shape= (8000, 4)))
    model.add(MaxPooling1D(pool_size=3, strides=3))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters, 10, activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=3))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters, 10, activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=3))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters, 10, activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=3))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters, 10, activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=3))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['mae'])

    filepath=plots_path+"model_"+str(layers)+"CNN_2LSTM_batch=256_f="+str(filters)+"_w="+str(window)+".best.hdf5"
    callbacks_list=[EarlyStopping(monitor='val_mean_absolute_error', patience=4),
                   ModelCheckpoint(filepath=filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True)]

    history=model.fit(x_train, y_train, batch_size=256, epochs=50, callbacks=callbacks_list,
                      verbose=0, validation_data=(x_val, y_val))

    # evaluate the model
    score_val=model.evaluate(x_val, y_val)
    print(f'\nValidation loss: {score_val[0]:.4} / MAE: {score_val[1]:.4}')
    df.loc[window, 'val_loss']=round(score_val[0],4)
    df.loc[window, 'val_mae']=round(score_val[1],4)

    # compute Pearson correlation coefficient with NumPy corrcoef function and write a report in dataframe
    y_new=model.predict(x_test)
    coeff=np.zeros(4)
    for i in range(4):
        coeff[i]=np.corrcoef(y_test[:, i], y_new[:, i])[0, 1]
    print("Pearson correlation coefficient: ", coeff)
    df.loc[window, 'PCC_1']=coeff[0]
    df.loc[window, 'PCC_2']=coeff[1]
    df.loc[window, 'PCC_3']=coeff[2]
    df.loc[window, 'PCC_4']=coeff[3]
    col_list_sum=['PCC_1', 'PCC_2', 'PCC_3', 'PCC_4']
    df['Sum']=df[col_list_sum].sum(axis=1)

    history_dict=history.history
    loss_values=history_dict['loss']
    val_loss_values=history_dict['val_loss']
    epochs=range(1, len(loss_values)+1)
    mae_values=history_dict['mean_absolute_error']
    val_mae_values=history_dict['val_mean_absolute_error']

    # plot a graph for loss values and save it
    fig_loss = plt.figure()
    plt.plot(epochs, loss_values, 'bo', label="Training loss")
    plt.plot(epochs, val_loss_values, 'b', label="Validation loss")
    plt.title("Training and validation LOSS")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    fig_loss.savefig(plots_path+'plot_loss_b=256_l='+str(layers)+'CNN_2LSTM_f='+str(filters)+'_w='+str(window)+'.png')

    # plot a graph for MAE values and save it
    fig_mae=plt.figure()
    plt.plot(epochs, mae_values, 'bo', label="Training MAE")
    plt.plot(epochs, val_mae_values, 'b', label="Validation MAE")
    plt.title("Training and validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend()
    plt.show()
    fig_mae.savefig(plots_path+'plot_mae_b=256_l='+str(layers)+'CNN_2LSTM_f='+str(filters)+'_w='+str(window)+'.png')

df.index.name='kernel_size'
max_index=df['Sum'].idxmax()
df.loc[max_index, 'Max']='MAX'
df=df.round(4)

print("\nDataFrame for a model with "+str(layers)+" Conv layers + 2LSTM batch_size=256 with "+str(filters)+" filters:\n", df)
df.to_csv(plots_path+"Dataframe_batch=256_l="+str(layers)+"Conv_2LSTM_f="+str(filters)+".csv")
