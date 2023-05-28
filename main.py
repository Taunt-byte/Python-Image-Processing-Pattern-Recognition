import cv2
import requests
import numpy as np
from io import BytesIO

# URL da imagem externa
url = "https://www.diariodoaco.com.br/images/noticias/84330/20201210080916_Iv6R16F2OT.jpg"

# Baixar a imagem da URL
response = requests.get(url)
imagem_bytes = response.content

# Converter os bytes em um objeto de imagem
imagem_array = np.array(bytearray(imagem_bytes), dtype=np.uint8)
imagem = cv2.imdecode(imagem_array, cv2.IMREAD_COLOR)

# Converter a imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Carregar o classificador pré-treinado
classificador = cv2.CascadeClassifier('caminho/para/classificador.xml')

# Detectar os padrões na imagem
padroes = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5)

# Desenhar retângulos nos padrões encontrados
for (x, y, w, h) in padroes:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Exibir a imagem com os padrões detectados
cv2.imshow('Padrões detectados', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
