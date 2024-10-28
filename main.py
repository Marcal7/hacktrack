import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

# Carregar o modelo pré-treinado
model = load_model('ia.h5')

# Dicionário de classes
classes = {0: 'Alicate', 1: 'cabo', 2: 'carrinho', 3:'chave de fenda', 4:'chave estrela', 5:'serrra'}

# Função para pré-processar a imagem
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Ajuste para o tamanho do modelo
    image = image / 255.0  # Normalizar para [0, 1]
    image = np.expand_dims(image, axis=0)  # Adicionar dimensão para batch
    return image

# Função para classificar a ferramenta
def classify_tool(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    tool_name = classes[class_id]
    confidence = prediction[0][class_id] * 100
    return tool_name, confidence


# Exemplo de uso
image_path = 'carrinho.png'
tool_name, confidence = classify_tool(image_path)


print(f"A ferramenta detectada é: {tool_name} com confiança de {confidence:.2f}%")
