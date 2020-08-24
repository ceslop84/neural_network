# def scale_data(data_frame):
#     # Padronização dos dados para preparar para a etapa de treinamento.
#     scaler = StandardScaler()
#     x_values = scaler.fit_transform(data_frame.drop('class', axis=1))
#     y_values = data_frame["class"].to_numpy()
#     array_scaled_merged = np.insert(x_values, 4, y_values, axis=1)
#     data_frame_scaled = pd.DataFrame(data=array_scaled_merged, columns=data_frame.columns)
#     return data_frame_scaled
    # Separação e preparação das variáveis para entrada no modelo.
    # df_train_sc = scale_data(df_train)
    # df_validate_sc = scale_data(df_validate)
    # df_test_sc = scale_data(df_test)
    # def train_input_fn():
    #     features = {'variance': np.array(df_train_sc["variance"]),
    #                 'skewness': np.array(df_train_sc["skewness"]),
    #                 'curtosis': np.array(df_train_sc["curtosis"]),
    #                 'entropy':  np.array(df_train_sc["entropy"])}
    #     labels = np.array(df_train_sc["class"])
    #     return features, labels

    # def val_input_fn():
    #     features = {'variance': np.array(df_validate_sc["variance"]),
    #                 'skewness': np.array(df_validate_sc["skewness"]),
    #                 'curtosis': np.array(df_validate_sc["curtosis"]),
    #                 'entropy':  np.array(df_validate_sc["entropy"])}
    #     labels = np.array(df_validate_sc["class"])
    #     return features, labels

    # def test_input_fn():
    #     features = {'variance': np.array(df_test_sc["variance"]),
    #                 'skewness': np.array(df_test_sc["skewness"]),
    #                 'curtosis': np.array(df_test_sc["curtosis"]),
    #                 'entropy':  np.array(df_test_sc["entropy"])}
    #     labels = np.array(df_test_sc["class"])
    #     return features, labels
    # feature_columns = [tf.feature_column.numeric_column("variance", dtype=tf.dtypes.float64),
    #                    tf.feature_column.numeric_column("skewness", dtype=tf.dtypes.float64),
    #                    tf.feature_column.numeric_column("curtosis", dtype=tf.dtypes.float64),
    #                    tf.feature_column.numeric_column("entropy", dtype=tf.dtypes.float64)]
    # classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10],
    #                                         n_classes=2,
    #                                         feature_columns=feature_columns)
    #train_res = classifier.train(input_fn=train_input_fn, max_steps=20)
    # val_res = classifier.evaluate(input_fn=val_input_fn)
    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**val_res))
    #test_res = list(classifier.predict(input_fn=test_input_fn))
#test_res = list(classifier.predict(input_fn=test_input_fn))
