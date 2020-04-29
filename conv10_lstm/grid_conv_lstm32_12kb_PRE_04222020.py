import h5py
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd

dataset="12000_PRE"
path="grid10/"
df=pd.DataFrame(columns=['val_loss', 'val_mae'])
np.set_printoptions(precision=4)

# read DNA sequences data
with h5py.File('seq_data_'+dataset+'.h5', 'r') as h5f:
    data = h5f['seq_data_'+dataset][:]
    h5f.close()

# read in the label data
with h5py.File('label_data.h5', 'r') as hf:
    label = hf['label_data'][:]
    hf.close()

# split dataset into training, validation and testing sets - 70/15/15
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
# split 30% of testing set into 15% testing and 15% validation
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=28)

def create_conv(layer_sizes):
    print("\nModel with layer_sizes: ", layer_sizes, "\n-------------------------------------------------")
    model = Sequential()
    model.add(Conv1D(filters=layer_sizes[0], kernel_size=10, activation='relu', input_shape= (x_train.shape[1], 4)))
    model.add(MaxPooling1D(pool_size=3, strides=3))

    for f in layer_sizes[1:]:
        model.add(Conv1D(filters=f, kernel_size=10, activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))
        model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(4, activation='softmax'))
    return model

def evaluate(model, layer_sizes, batch_size=256, epochs=50):
    model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['mae'])

    filepath=path+"model_data="+dataset+"_"+str(len(layer_sizes))+"_conv_layers_lstm32_f=_"+str(layer_sizes[0])+".best.hdf5"
    callbacks_list=[EarlyStopping(monitor='val_mean_absolute_error', patience=4),
                   ModelCheckpoint(filepath=filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True)]

    history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                      callbacks=callbacks_list,
                      verbose=0, validation_data=(x_val, y_val))

    score_val=model.evaluate(x_val, y_val, verbose=0)
    print(f'\nValidation loss: {score_val[0]:.4} / MAE: {score_val[1]:.4}')
    df.loc[str(layer_sizes), 'val_loss']=round(score_val[0],4)
    df.loc[str(layer_sizes), 'val_mae']=round(score_val[1],4)

    # compute Pearson correlation coefficient with NumPy corrcoef function
    y_new=model.predict(x_test)
    coeff=np.zeros(4)
    for i in range(4):
        coeff[i]=np.corrcoef(y_test[:, i], y_new[:, i])[0, 1]
    print("Pearson correlation coefficient: ", coeff, "Sum: ", np.sum(coeff))
    df.loc[str(layer_sizes), 'PCC_1']=coeff[0]
    df.loc[str(layer_sizes), 'PCC_2']=coeff[1]
    df.loc[str(layer_sizes), 'PCC_3']=coeff[2]
    df.loc[str(layer_sizes), 'PCC_4']=coeff[3]
    col_list_sum=['PCC_1', 'PCC_2', 'PCC_3', 'PCC_4']
    df['Sum']=df[col_list_sum].sum(axis=1)


# 3 layers each with 32 output units, then 4 layers each with 128 output units etc.
for filters_per_layer in [32, 64, 128]:
    for layers in [2, 3, 4, 5, 6]:
        layer_sizes=[filters_per_layer] * layers
        model = create_conv(layer_sizes)
        evaluate(model, layer_sizes)

df.index.name='layer_sizes'
max_index=df['Sum'].idxmax()
df.loc[max_index, 'Max']='MAX'
df=df.round(4)

print("\nDataFrame for a model with grid Conv1D layers_LSTM32 for "+dataset+" dataset:\n", df)
df.to_csv(path+"Dataframe_grid_conv_lstm32_"+dataset+".csv")
