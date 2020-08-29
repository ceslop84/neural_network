import pandas as pd
#import tensorflow (conda install -c conda-forge tensorflow)
from tensorflow.contrib.layers import real_valued_column
from tensorflow.contrib.learn import DNNClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from data import read_data

# VERIFICAR SE EXISTE COMO SETAR O LIMIAR. PROF CITOU EM CLASSE

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
    feature_columns = [real_valued_column("", dimension=4)]
    classifier = DNNClassifier(hidden_units=[10, 20, 10],
                               n_classes=2,
                               feature_columns=feature_columns)
    # Treinamento.
    classifier.fit(x_train, y_train, steps=200, batch_size=20)
    # Avaliação.
    val_res = classifier.evaluate(x_val, y_val)

    # Teste.
    test_res = list(classifier.predict(x_test))
    print('Acurácia validação modelo \n {accuracy:0.3f}'.format(**val_res))
    print('Matriz de confusão \n', confusion_matrix(y_test, test_res))
    print('Relatório de classificação \n', classification_report(y_test, test_res))
    print('Acurácia teste modelo \n', accuracy_score(y_test, test_res))

if __name__ == "__main__":
    # Leitura dos dados de entrada.
    ARQUIVO = "entrada.txt"
    # Execução do método principal.
    aut_bancaria_rn(ARQUIVO)
