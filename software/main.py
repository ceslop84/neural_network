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

def aplicar_rn(x_test, y_test, modelo):
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
    return pco
    # print(str(round(pco, 2)))
    # # # Geração do arquivo de saída.
    # saida_df = pd.DataFrame(lista)
    # saida_df.columns = ["variance", "skewness",
    #                     "curtosis", "entropy", "class", "result"]
    # # Registro da hora de início para a geração dos arquivos de saída em pasta específica.
    # timestamp = str(datetime.today().strftime('%Y%m%d_%H%M%S'))
    # saida_df.to_csv(timestamp + "_" + nome + ".csv", index=False)

def aut_bancaria_rn(arquivo, modelo, loss, optimizer, metrics, batch_size, epochs):
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

    modelo.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=metrics)

    # Treinamento.
    modelo.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val))
    # Avaliação.
    # val_res = model.evaluate(x_test, y_test, batch_size=20)
    pco_val = aplicar_rn(x_val, y_val, modelo)
    # Teste.
    pco_test = aplicar_rn(x_test, y_test, modelo)
    return [pco_val, pco_test]

def criar_cenarios():

    modelo = list()
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+o_1_sigmoid"])
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+10_relu+o_1_sigmoid"])
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+10_relu+10_relu+o_1_sigmoid"])
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+10_relu+10_relu+10_relu+o_1_sigmoid"])
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+10_relu+10_relu+10_relu+10_relu+o_1_sigmoid"])
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(10, activation='tanh'),
                                     keras.layers.Dense(20, activation='tanh'),
                                     keras.layers.Dense(10, activation='tanh'),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+10_tanh+20_tanh+10_tanh+o_1_sigmoid"])
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(20, activation='relu'),
                                     keras.layers.Dense(10, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+10_relu+20_relu+10_relu+o_1_sigmoid"])
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(10, activation='linear'),
                                     keras.layers.Dense(20, activation='linear'),
                                     keras.layers.Dense(10, activation='linear'),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+10_linear+20_linear+10_linear+o_1_sigmoid"])
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(100, activation='relu'),
                                     keras.layers.Dense(200, activation='relu'),
                                     keras.layers.Dense(100, activation='relu'),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+100_relu+200_relu+100_relu+o_1_sigmoid"])
    modelo.append([keras.Sequential([keras.Input(shape=(4,)),
                                     keras.layers.Dense(10, activation='tanh'),
                                     keras.layers.Dense(20, activation='relu'),
                                     keras.layers.Dense(10, activation='linear'),
                                     keras.layers.Dense(1, activation='sigmoid')]),
                                     "i_4+10_tanh+20_relu+10_linear+o_1_sigmoid"])

    loss = list()
    loss.append("binary_crossentropy")

    optimizer = list()
    optimizer.append("sgd")
    optimizer.append("Adam")
    optimizer.append("RMSprop")

    metrics = list()
    # metrics.append(["accuracy"])
    metrics.append(["binary_accuracy"])

    batch_size = list()
    batch_size.append(4)
    batch_size.append(16)
    batch_size.append(32)

    epochs = list()
    epochs.append(5)
    epochs.append(50)
    epochs.append(200)

    return modelo, loss, optimizer, metrics, batch_size, epochs

if __name__ == "__main__":
    # Leitura dos dados de entrada.
    ARQUIVO = "entrada.txt"

    #Criação dos cenários de testes.
    modelo, loss, optimizer, metrics, batch_size, epochs = criar_cenarios()

    # Execução do método principal.
    total = len(modelo)*len(loss)*len(optimizer)*len(metrics)*len(batch_size)*len(epochs)
    i=0
    resultados = list()
    for md in modelo:
        for l in loss:
            for o in optimizer:
                for mt in metrics:
                    for b in batch_size:
                        for e in epochs:
                            print("")
                            print(f"Calculando iteração {i} de um total de {total}...")
                            print("")
                            res = aut_bancaria_rn(ARQUIVO,
                                                modelo=md[0],
                                                loss=l,
                                                optimizer=o,
                                                metrics=mt,
                                                batch_size=b,
                                                epochs=e)
                            res.append(md[1])
                            res.append(str(l))
                            res.append(str(o))
                            res.append(str(mt))
                            res.append(str(b))
                            res.append(str(e))
                            resultados.append(res)
                            i+=1
    # Geração do arquivo de saída.
    saida_df = pd.DataFrame(resultados)
    saida_df.columns = ["pco_val",
                        "pco_test",
                        "modelo",
                        "loss",
                        "optimizer",
                        "metrics",
                        "batch_size",
                        "epochs"]
    # Registro da hora de início para a geração dos arquivos de saída em pasta específica.
    timestamp = str(datetime.today().strftime('%Y%m%d_%H%M%S'))
    saida_df.to_csv(timestamp + ".csv", index=False)
