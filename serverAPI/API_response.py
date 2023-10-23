#Instalaciones necesarias para el google colab 

!pip install transformers
!pip install transformers --upgrade
!pip install mtranslate
!pip install torch
!pip install langdetect
!pip install flask
!pip install flask_cors
 
!pip install pyngrok

 
!ngrok authtoken 2QqsUKuyaix37IWHno2HikaVqrf_5C8Up2BuMbsdhAzZAEhjP
# Importar las bibliotecas necesarias
from pyngrok import ngrok
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mtranslate import translate
import torch
from langdetect import detect
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
 
# Crear una instancia de la aplicación Flask para crear un servidor
app = Flask(__name__)
 
# Habilitar el CORS (Compartir Recursos de Origen Cruzado) para permitir solicitudes desde cualquier origen
CORS(app, resources={r"/*": {"origins": "*"}})
 
#Definir la ruta del endpoint "/generate-response/<question>" para generar la respuesta recibiendo en la url la pregunta como "question"
@app.route('/generate-response/<question>', methods=['POST'])
def generate_response(question):
  # Obtener la pregunta del cuerpo de la solicitud en formato JSON
    question = request.json['question']
 
    # Verificar si hay una GPU disponible para utilizarla, de lo contrario utilizar la CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Detectar el idioma de la pregunta
    source_lang = detect(question)
 
    # Traducir la pregunta al inglés si no está en inglés
    if source_lang != "en":
        question = translate(question, "en")
 
    # Carga el tokenizador preentrenado GPT2 (Para que el modelo pueda entender la pregunta)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                  "bos_token": "<startofstring>",
                                  "eos_token": "<endofstring>"})
    tokenizer.add_tokens(["<bot>:"])
 
    # Carga el modelo de GPT2
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    #Carga el estado del modelo preentrenado con las bases de datos de piscologia al modelo de gpt2
    model.load_state_dict(torch.load("/content/drive/MyDrive/TFG-FINAL/pyschoGPT.pt", map_location=device))
    model.to(device)
    model.eval()
 
    # Preprocesar la pregunta agregando los tokens especiales para que el modelo pueda entender la pregunta
    input_text = "<startofstring> " + question + " <bot>: "
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
 
    # Generar la respuesta utilizando el modelo
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
 
    # Traducir la respuesta al idioma original de la pregunta si no está en inglés
    if source_lang != "en":
        output_text = translate(output_text, source_lang, "en")
 
    # Obtener el texto generado por el modelo eliminando el texto anterior a "<bot>:" y posterior a "<bot>:"
    start_token = "<bot>:"
    end_token = "<bot>:"
    start_index = output_text.find(start_token)
    end_index = output_text.find(end_token, start_index + len(start_token))
    if start_index != -1 and end_index != -1:
        output_text = output_text[start_index + len(start_token):end_index].strip()
 
    # Devolver la respuesta generada como JSON
    return jsonify({'response': output_text})
 
if __name__ == '__main__':
    # Obtener el puerto aleatorio disponible en el entorno o utilizar el puerto 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
 
    # Iniciar el túnel de Ngrok para crear una URL pública y accesible desde Internet
    # Este paso es necesario ya que desde google colab no se puede hacer el server flask ya que lo bloquea entonces usamos un intermediario.
    ngrok_tunnel = ngrok.connect(port)
    print('Public URL:', ngrok_tunnel.public_url)
 
    # Ejecutar la aplicación Flask utilizando la URL proporcionada por Ngrok
    app.run(host='0.0.0.0', port=port)
