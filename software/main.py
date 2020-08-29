"""Módulo principal para calcular a autencidade de uma nota através de técnicas de Redes Neurais."""
from datetime import datetime
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from data import read_data


def scale_data_xy(data_frame):
    """ Método para padronizar os dados previamente ao treinamento.

    Params:
        data_frame (DataFrame): Objeto com os dados de entrada.

    Returns:
        Array: Valores referentes à função X.
        Array: Valores referentes à função Y.
    """
    # Padronização dos dados para preparar para a etapa de treinamento.
    scaler = StandardScaler()
    x_sc = scaler.fit_transform(data_frame.drop('class', axis=1))
    df_x_sc = pd.DataFrame(data=x_sc, columns=data_frame.columns[:-1])
    x_values = df_x_sc.to_numpy()
    y_values = data_frame["class"].to_numpy()
    return x_values, y_values

def aplicar_rn(nome, x_test, y_test, modelo):
    """ Método para aplicar as regras fuzzy criadas anteriormente num determinado
    conjunto de dados.

    Params:
        nome (str): Nome do conjunto de dados.
        x_test (Array): Objeto contendo os dados a serem aplicados a rede neural.
        y_test (Array): Objeto contendo os resultados esperados da rede neural.
        modelo (Model): Classe para a aplicação da inferênia de redes neurais sobre os dados.
    """
    # Aplicação do método e regras para o conjunto de validação.
    rn_res = modelo.predict(x_test)
    lista = list()
    total = 0
    total_correto = 0
    i = 0
    for res in rn_res:
        y_res = round(res[0], 0)
        elem = list()
        elem.append(x_test[i, 0])
        elem.append(x_test[i, 1])
        elem.append(x_test[i, 2])
        elem.append(x_test[i, 3])
        elem.append(y_test[i])
        elem.append(y_res)
        lista.append(elem)
        if y_test[i] == y_res:
            total_correto += 1
        total += 1
        i += 1
    # # Cálculo e impressão do Percentual de Classe Correta - PCO.
    pco = 100 * (total_correto/total)
    print(str(round(pco, 2)))
    # # Geração do arquivo de saída.
    saida_df = pd.DataFrame(lista)
    saida_df.columns = ["variance", "skewness",
                        "curtosis", "entropy", "class", "result"]
    # Registro da hora de início para a geração dos arquivos de saída em pasta específica.
    timestamp = str(datetime.today().strftime('%Y%m%d_%H%M%S'))
    saida_df.to_csv(timestamp + "_" + nome + ".csv", index=False)

def aut_bancaria_rn(arquivo):
    """ Método que detecta autenticidade de notas bancárias.
        Este método usa redes neurais para realizar tal avaliação.

    Params:
        arquivo (str): Nome do arquivo com os dados de entrada.
    """
    # Leitura do arquivo com os dados de entrada e separação em treinamento, validação e teste.
    df_train, df_val, df_test = read_data(arquivo)
    x_train, y_train = scale_data_xy(df_train)
    x_val, y_val = scale_data_xy(df_val)
    x_test, y_test = scale_data_xy(df_test)

    # Criação da rede neural.
    model = keras.Sequential([keras.Input(shape=(4,)),
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
              epochs=500,
              validation_data=(x_val, y_val))
    # Avaliação.
    # val_res = model.evaluate(x_test, y_test, batch_size=20)
    aplicar_rn("validação",
               x_val,
               y_val,
               model)
    # Teste.
    aplicar_rn("teste",
               x_test,
               y_test,
               model)

if __name__ == "__main__":
    # Leitura dos dados de entrada.
    ARQUIVO = "entrada.txt"
    # Execução do método principal.
    aut_bancaria_rn(ARQUIVO)
