from flask import Flask, request, jsonify
from flask_cors import CORS # Importe o CORS
from modelo import treinar_modelo, prever_sobrevivencia

app = Flask(__name__)
CORS(app) # Habilita o CORS para todas as rotas

@app.route('/')
def home():
    return "API do Modelo Titanic"

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    test_size = data.get('test_size', 0.2)
    activation_function = data.get('activation_function', 'relu')
    optimiser = data.get('optimiser', 'adam')
    loss_function = data.get('loss_function', 'binary_crossentropy')
    epochs = data.get('epochs', 100)
    batch_size = data.get('batch_size', 32)
    
    # NOVOS PARAMETROS:
    num_hidden_layers = data.get('num_hidden_layers', 1)
    neurons_per_layer = data.get('neurons_per_layer', 32)
    dropout_rate = data.get('dropout_rate', 0.0)


    # Passe os novos parâmetros para a função de treino
    results = treinar_modelo(test_size, activation_function, optimiser, loss_function, epochs, batch_size,
                             num_hidden_layers, neurons_per_layer, dropout_rate)
    return jsonify(results)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Assegurar que todos os campos esperados estão presentes
    expected_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for field in expected_fields:
        if field not in data:
            return jsonify({"error": f"Campo '{field}' faltando na requisição."}), 400

    # A função prever_sobrevivencia já espera um dicionário com esses campos
    prediction_result = prever_sobrevivencia(data)
    
    # prever_sobrevivencia pode retornar um erro, verificar isso
    if isinstance(prediction_result, tuple): # Se for um tuple, é erro (resultado, status_code)
        return jsonify(prediction_result[0]), prediction_result[1]
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
