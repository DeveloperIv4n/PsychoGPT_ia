#en google colab se usa esta estructura para instalarlos
!pip install transformers
!pip install transformers --upgrade
#imports para las librerias que vamos a utilizar
from torch.utils.data import Dataset
import json


class ChatData(Dataset):
    def __init__(self,path:str,tokenizer):
      # Carga los datos desde el archivo JSON
        self.data = json.load(open(path,"r"))

        self.X = []
        for i in self.data:
          # Obtiene los textos de entrada y salida del chat
            self.X.append(i['input'])
            self.X.append(i['output'])
            
          # Concatena los textos de entrada y salida en un solo texto,
          # agrega marcas especiales para indicar el inicio y el final del texto,
          # y lo almacena en la lista self.X
        for idx,i in enumerate(self.X):
            try:
                self.X[idx] = "<startofstring> "+i+" <bot>: "+self.X[idx+1]+" <endofstring>"
            except:
                break
        
        self.X = self.X[:-1]   
        # Codifica los textos utilizando el tokenizer proporcionado
        self.X_encoded = tokenizer(self.X,max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
      # Devuelve la longitud de los datos
        return len(self.X) 

    def __getitem__(self,idx):
       # Devuelve un par de tensores que representan el input_ids y la atención para un ejemplo en particular
        return (self.input_ids[idx],self.attention_mask[idx])