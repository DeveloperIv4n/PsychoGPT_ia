import streamlit as st
import torch
from mtranslate import translate
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langdetect import detect
import gdown
def generate_response(question):
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
    url = 'https://drive.google.com/uc?id=1YKCC8flDS3rL8g1aduOpp00IwWO223hR'
    output_model = 'pyschoGPT.pt'
    gdown.download(url, output_model, quiet=False)

    model.load_state_dict(torch.load(output_model, map_location=device))
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
 
    # Devolver la respuesta generada
    return output_text

# Streamlit UI
st.title("Pregunta a PsychoGPT")

# Recoger la pregunta del usuario
question = st.text_input("Haz una pregunta")

if st.button("Generar respuesta"):
    # Generar la respuesta
    response = generate_response(question)
    st.write(" ", response)
