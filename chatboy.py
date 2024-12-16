# Importación de las librerías necesarias
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import random

# Descargar recursos de NLTK para preprocesamiento (opcional)
nltk.download('punkt')
nltk.download('stopwords')

# Datos de entrenamiento (frases de entrada y etiquetas asociadas)
training_sentences = [
    "¿Cuánto cuesta el producto?",
    "¿Tienes camisas rojas?",
    "¿Dónde está mi pedido?",
    "¿Cómo puedo pagar?",
    "¿Cuál es el horario de atención?",
    "Quiero saber el precio del artículo",
    "¿Cómo puedo realizar el pago?",
    "Quiero saber si tienes camisas rojas en stock",
    "¿A qué hora abren?",
    "Mi pedido no ha llegado",
    "¿Puedo pagar con tarjeta?",
    "¿Dónde está mi compra?"
]

training_labels = [
    "precio",  # Pregunta sobre precio
    "producto",  # Pregunta sobre productos
    "pedido",  # Pregunta sobre el estado de un pedido
    "pago",  # Pregunta sobre opciones de pago
    "horario",  # Pregunta sobre el horario
    "precio",  # Pregunta sobre precio
    "pago",  # Pregunta sobre pago
    "producto",  # Pregunta sobre productos
    "horario",  # Pregunta sobre horario
    "pedido",  # Pregunta sobre pedido
    "pago",  # Pregunta sobre pago
    "pedido"  # Pregunta sobre pedido
]

# Paso 1: Preprocesamiento de texto
# Vectorización de las frases para convertirlas en vectores numéricos
vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('spanish'))
X_train = vectorizer.fit_transform(training_sentences)

# Paso 2: Entrenamiento del modelo
# Usamos el clasificador Naive Bayes para clasificación
classifier = MultinomialNB()
classifier.fit(X_train, training_labels)

# Paso 3: Función para predecir la intención de una nueva entrada
def predict_intent(user_input):
    # Preprocesar la entrada del usuario (convertir en vector numérico)
    input_vector = vectorizer.transform([user_input])
    # Predecir la clase (intención) usando el modelo entrenado
    prediction = classifier.predict(input_vector)
    return prediction[0]

# Paso 4: Base de respuestas predefinidas
responses = {
    "precio": ["El precio del producto es $30.", "El artículo cuesta $30.", "Este producto tiene un costo de $30."],
    "producto": ["Sí, tenemos camisas rojas disponibles.", "Sí, las camisas rojas están en stock.", "Las camisas rojas están disponibles en tallas M y L."],
    "pedido": ["Su pedido está en camino y llegará mañana.", "Tu pedido será entregado mañana.", "El pedido ya fue enviado y debería llegar pronto."],
    "pago": ["Aceptamos pagos con tarjeta de crédito, débito y PayPal.", "Puedes pagar con tarjeta o PayPal.", "Aceptamos tarjetas Visa, MasterCard y pagos por PayPal."],
    "horario": ["Nuestro horario de atención es de 9:00 AM a 6:00 PM.", "Estamos disponibles de lunes a viernes, de 9 AM a 6 PM.", "Abrimos de 9:00 AM a 6:00 PM de lunes a viernes."]
}

# Paso 5: Interacción con el usuario
print("Bienvenido al Chatbot de la tienda en línea. ¿Cómo puedo ayudarte hoy?")
while True:
    user_input = input("Usuario: ")
    # Predecir la intención de la entrada del usuario
    intent = predict_intent(user_input)
    
    # Responder según la intención predicha
    if intent in responses:
        print(f"Chatbot: {random.choice(responses[intent])}")
    else:
        print("Chatbot: Lo siento, no entendí esa pregunta. ¿Puedes reformularla?")
