import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pickle

# Função para carregar e pré-processar os dados
def carregar_e_preparar_dados():
    # Carregar o dataset
    try:
        # CORREÇÃO: Alterado de 'titanic.csv' para 'train.csv'
        df = pd.read_csv('train.csv') 
    except FileNotFoundError:
        # CORREÇÃO: Mensagem de erro atualizada
        print("Erro: O arquivo 'train.csv' não foi encontrado. Certifique-se de que ele está na mesma pasta do script.")
        # Criar um DataFrame de exemplo para evitar quebra total, se necessário
        data = {
            'PassengerId': range(1, 11),
            'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
            'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath (Lily May Peel)', 'Allen, Mr. William Henry', 'Moran, Mr. James', 'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard', 'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)', 'Nasser, Mrs. Nicholas (Adele Achem)'],
            'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
            'Age': [22, 38, 26, 35, 35, np.nan, 54, 2, 27, 14],
            'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
            'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '330877', '17463', '349909', '347742', '237736'],
            'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708],
            'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan, np.nan, 'E46', np.nan, np.nan, np.nan],
            'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C']
        }
        df = pd.DataFrame(data)
        print("Usando DataFrame de exemplo. Por favor, adicione 'train.csv' para dados reais.")


    # Preencher valores ausentes
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True) # Raramente Fare tem NA, mas para robustez

    # Remover colunas que não serão usadas
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Definir features (X) e target (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Definir colunas numéricas e categóricas para pré-processamento
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    # Criar pré-processador usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Manter outras colunas (se houver, o que não é o caso aqui)
    )

    return X, y, preprocessor

# Função para treinar o modelo
def treinar_modelo(test_size, activation_function, optimiser, loss_function, epochs, batch_size, 
                   num_hidden_layers=1, neurons_per_layer=32, dropout_rate=0.0):
    
    X, y, preprocessor = carregar_e_preparar_dados()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Ajustar o pré-processador aos dados de treino
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Obter o número de features após o OneHotEncoding para a camada de entrada
    input_shape = X_train_processed.shape[1]

    # Construir o modelo Keras dinamicamente
    model = Sequential()
    
    # Primeira camada oculta
    model.add(Dense(neurons_per_layer, activation=activation_function, input_shape=(input_shape,)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Camadas ocultas adicionais
    for _ in range(num_hidden_layers - 1): # -1 porque a primeira já foi adicionada
        model.add(Dense(neurons_per_layer, activation=activation_function))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Camada de saída
    model.add(Dense(1, activation='sigmoid')) # Saída binária para classificação (sobreviveu/não)

    model.compile(optimizer=optimiser, loss=loss_function, metrics=['accuracy'])

    history = model.fit(X_train_processed, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test_processed, y_test),
                        verbose=0) # verbose=0 para não printar cada época

    # Salvar o modelo e o scaler
    model.save('titanic_model.h5')
    with open('scaler_titanic.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    final_val_accuracy = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]

    return {"message": "Modelo treinado e salvo com sucesso!", 
            "val_accuracy": final_val_accuracy, 
            "val_loss": final_val_loss}

# Função para fazer a previsão
def prever_sobrevivencia(data):
    try:
        model = load_model('titanic_model.h5')
        with open('scaler_titanic.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except (OSError, FileNotFoundError) as e:
        return {"error": f"Modelo ou scaler não encontrados. Por favor, treine o modelo primeiro. Erro: {e}"}, 500

    # Criar um DataFrame a partir dos dados de entrada
    input_df = pd.DataFrame([data])

    # Garantir que as colunas estejam na mesma ordem que no treinamento
    expected_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for col in expected_cols:
        if col not in input_df.columns:
            return {"error": f"Dados de entrada incompletos: '{col}' está faltando."}, 400
    
    # Reordenar colunas para garantir consistência
    input_df = input_df[expected_cols]

    # Aplicar o pré-processador
    try:
        input_processed = preprocessor.transform(input_df)
    except Exception as e:
        return {"error": f"Erro ao pré-processar os dados de entrada: {e}. Verifique se os tipos de dados estão corretos."}, 400


    # Fazer a previsão
    prediction_proba = model.predict(input_processed)[0][0]
    prediction = (prediction_proba > 0.5).astype(int)

    # Mensagem personalizada
    if prediction == 1:
        message = "Sobreviveu!"
    else:
        message = "Não sobreviveu."

    return {
        "previsao": prediction.item(), # .item() para converter numpy int para python int
        "probabilidade_sobreviver": prediction_proba.item(), # .item() para float
        "mensagem": message
    }
