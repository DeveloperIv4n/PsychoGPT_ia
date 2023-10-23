from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

# Importación de las bibliotecas necesarias

def train(chatData, model, optim):
    # Función de entrenamiento del modelo
    epochs = 12
    # Número de épocas de entrenamiento 
    #Cuantas mas epocas o epochs mas preciso será pero más recursos computacionales consume, puede ocurrir el sobreajuste (0 precisión en la respuesta)

    for i in tqdm.tqdm(range(epochs)):
        # For para a recorrer las épocas(epochs) utilizando la libreria tqdm para mostrar una barra de progreso
        for X, a in chatData:
            X = X.to(device)
            a = a.to(device)
            # Mover los datos a la GPU si está disponible ya que por GPU es mas eficiente
            optim.zero_grad()
            # Reiniciar los gradientes acumulados
            loss = model(X, attention_mask=a, labels=X).loss
            # Calcular la pérdida del modelo, la perdida sirve para monitorizar y ver la precisión de las respuestas
            loss.backward()
            # Retropropagación de la pérdida para calcular los gradientes. Este paso es esencial para poder ajustar los pesos del modelo y mejorar su rendimiento.
            optim.step()
            # Actualización de los pesos del modelo utilizando el optimizador

        torch.save(model.state_dict(), "/content/drive/MyDrive/IA/model_state/pyschoGPT.pt")
        model.config.save_pretrained("/content/drive/MyDrive/IA/model_config/")
        tokenizer.save_pretrained("/content/drive/MyDrive/IA/tokenizer_info/")
        # Guardar el estado del modelo y los archivos relacionados como la configuracion del modelo y su tokenizador.

        print(infer("I'm having trouble sleeping and it's affecting my daily life."))
        print(infer("I'm having trouble with my child's behavior and I don't know how to handle it."))
        print(infer("I'm struggling with my finances and don't know how to manage my debt."))
        print(infer("I'm having trouble with my time management and I don't know how to improve it."))
        # Ejecutar una funcion que va generando respuestas a estas preguntas para ver como va mejorando la calidad de las respuestas según se entrena

# Función de inferencia utilizando el modelo entrenado
def infer(inp):
    inp = "<startofstring> " + inp + " <bot>: "
    inp = tokenizer.encode(inp, return_tensors="pt")
    inp = inp.to(device)
    output = model.generate(inp, max_length=100, num_return_sequences=1)
    output = tokenizer.decode(output[0])
    return output
device = "cuda" if torch.cuda.is_available() else "cpu"
# Verificar la disponibilidad de GPU para aceleración de cómputo - Esto sirve para el google colab debido a que tiene recursos limitados

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Cargar el tokenizador GPT-2 preentrenado previamente almacenado

tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])
# Añadir tokens especiales al tokenizador para distinguir la pregunta de la respuesta 

model = GPT2LMHeadModel.from_pretrained("gpt2")
# Cargar el modelo GPT-2 preentrenado de huggingface

model.resize_token_embeddings(len(tokenizer))
# Ajustar la dimensión del espacio de embeddings del modelo al tamaño del tokenizador

model = model.to(device)
# Mover el modelo a la GPU si está disponible ya que como dije anteriormente es mas eficiente

chatData = ChatData("/content/drive/MyDrive/IA/psychology.json", tokenizer)
# Almacenar los datos preporcesados gracias a la funcion para usarlos para entrenar al modelo

chatData = DataLoader(chatData, batch_size=128)
# Crear un DataLoader para cargar los datos de chat en lotes de tamaño 128

model.train()
# Establecer el modelo en modo de entrenamiento

optim = Adam(model.parameters(), lr=1e-4)
# Crear el optimizador Adam para ajustar los pesos del modelo con una tasa de aprendizaje de 1e-4 , cuanto mas pequeña es la medida mas preciso es pero mas recursos consume para entrenarlo

print("training .... ")
train(chatData, model, optim)
#Entrenar el modelo
print("Escribir pregunta : ")
while True:
  inp = input()
  print(infer(inp))
#while que utilicé para testear el modelo