import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Inicialización de la cámara, el detector de manos y el clasificador
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Parámetros y etiquetas
offset = 20
imgSize = 300
labels = ["Bien", "Chao", "Gracias", "Hola", "Mal", "No", "Perdon", "Si", "Te Amo"]

while True:
    # Capturar fotograma de la cámara
    success, img = cap.read()
    imgOutput = img.copy()

    # Detectar la mano en la imagen
    hands, img = detector.findHands(img)

    if hands:
        # Obtener la información de la mano
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crear una imagen en blanco para el recorte
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Recortar la región de la mano
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Redimensionar la región recortada para que sea cuadrada
        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))

        # Actualizar la imagen en blanco con la región redimensionada
        imgWhite[:imgSize, :imgSize] = imgResize

        # Obtener la predicción del clasificador
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Ajustar el umbral aquí
        threshold = 0.7
        if prediction[index] > threshold:
            # Mostrar resultados en la imagen de salida
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                          cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Mostrar imágenes intermedias (opcional)
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Mostrar la imagen de salida
    cv2.imshow('Image', imgOutput)

    # Comprobar si se ha presionado la tecla Esc para salir
    key = cv2.waitKey(1)
    if key == 27:
        break

# Liberar recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

    