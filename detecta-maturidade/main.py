import numpy as np
import cv2
import torch
from torchvision import models, transforms
from PIL import Image

# Caminhos dos arquivos
ARQUIVO_VIDEO = "./melancia.mp4"  # Caminho do vídeo
ARQUIVO_MODELO = 'fruits_vegetables_51.pth'  # Caminho do modelo treinado (ex: ResNet-50)

# Definir o modelo
model = models.resnet50(weights=None)
num_classes = 51  # Ajustar para o número de classes do modelo treinado
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(ARQUIVO_MODELO, weights_only=True))
model.eval()

# Definir as transformações de imagem
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classes do modelo (ajustadas para o Fruit 360)
class_names = [
    'Amaranth', 'Apple', 'Banana', 'Beetroot', 'Bell pepper', 'Bitter Gourd', 'Blueberry', 'Bottle Gourd',
    'Broccoli', 'Cabbage', 'Cantaloupe', 'Capsicum', 'Carrot', 'Cauliflower', 'Chilli pepper', 'Coconut',
    'Corn', 'Cucumber', 'Dragon_fruit', 'Eggplant', 'Fig', 'Garlic', 'Ginger', 'Grapes', 'Jalepeno',
    'Kiwi', 'Lemon', 'Mango', 'Okra', 'Onion', 'Orange', 'Paprika', 'Pear', 'Peas', 'Pineapple',
    'Pomegranate', 'Potato', 'Pumpkin', 'Raddish', 'Raspberry', 'Ridge Gourd', 'Soy beans', 'Spinach',
    'Spiny Gourd', 'Sponge Gourd', 'Strawberry', 'Sweetcorn', 'Sweetpotato', 'Tomato', 'Turnip', 'Watermelon'
]

# Função para detecção de frutas em uma imagem
def detectar_fruta(frame):
    # Converte a imagem do OpenCV para PIL para o modelo PyTorch
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(pil_image).unsqueeze(0)

    # Move para o dispositivo correto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model.to(device)

    # Realiza a previsão
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    # Retorna a classe prevista
    return class_names[predicted_class.item()]

def main():
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    pausado = False
    fruta_inicial = None
    fruta_inicial_detectada = False

    while True:
        if not pausado:
            ret, frame = captura.read()
            if not ret:
                break

            # Realiza a detecção de frutas e vegetais no frame
            fruta_detectada = detectar_fruta(frame)

            if not fruta_inicial_detectada:
                # Armazena a fruta detectada no primeiro quadro
                fruta_inicial = fruta_detectada
                fruta_inicial_detectada = True
                status = fruta_inicial  # Fruta que foi detectada inicialmente
            else:
                if fruta_detectada != fruta_inicial:
                    # Se a fruta mudou, consideramos podre
                    status = "Podre"
                else:
                    status = fruta_inicial  # Mantém a fruta inicial se não mudou

            # Exibe o nome da fruta detectada ou 'Podre'
            cv2.putText(frame, f"{status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibe o frame processado
        cv2.imshow("Detecção de Frutas e Vegetais", frame)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("p"):
            pausado = not pausado

    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
