import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from data import read_data


def scale_data_xy(data_frame):
    # Padronização dos dados para preparar para a etapa de treinamento.
    scaler = StandardScaler()
    x_sc = scaler.fit_transform(data_frame.drop('class', axis=1))
    df_x_sc = pd.DataFrame(data=x_sc, columns=data_frame.columns[:-1])
    x_values = df_x_sc.to_numpy()
    y_values = data_frame["class"].to_numpy()
    return x_values, y_values

def aut_bancaria_rn(arquivo):
    # Leitura do arquivo com os dados de entrada e separação em treinamento, validação e teste.
    df_train, df_val, df_test = read_data(arquivo)
    x_train, y_train = scale_data_xy(df_train)
    x_val, y_val = scale_data_xy(df_val)
    x_test, y_test = scale_data_xy(df_test)

    # Criação da rede neural.
    # inputs = keras.Input(shape=(4,), name="image_data")
    # x = layers.Dense(10, activation="relu", name="dense_1")(inputs)
    # x = layers.Dense(20, activation="relu", name="dense_2")(x)
    # x = layers.Dense(10, activation="relu", name="dense_3")(x)
    # outputs = layers.Dense(1, activation="softmax", name="predictions")(x)
    # model = keras.Model(inputs=inputs, outputs=outputs)

    model = keras.Sequential([
                              keras.Input(shape=(4,)),
                              keras.layers.Dense(10, activation='tanh'),
                              keras.layers.Dense(20, activation='tanh'),
                              keras.layers.Dense(10, activation='tanh'),
                              keras.layers.Dense(1, activation='sigmoid')])


    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    # Treinamento.
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=100,
              validation_data=(x_val, y_val))

    val_res = model.evaluate(x_test, y_test, batch_size=20)
    #print("test loss, test acc:", results)
    test_res = model.predict(x_test)
    #print("predictions shape:", predictions.shape)
    #print('Acurácia validação modelo \n {accuracy:0.3f}'.format(**val_res))
    print("\nLoss, accuracy on test data: ")
    print("%0.4f %0.2f%%" % (val_res[0], val_res[1]*100))
    print('Matriz de confusão \n', confusion_matrix(y_test, test_res))
    print('Relatório de classificação \n', classification_report(y_test, test_res))
    print('Acurácia teste modelo \n', accuracy_score(y_test, test_res))

if __name__ == "__main__":
    # Leitura dos dados de entrada.
    ARQUIVO = "entrada.txt"
    # Execução do método principal.
    aut_bancaria_rn(ARQUIVO)
