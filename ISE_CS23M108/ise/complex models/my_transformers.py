# Install transformers library if not already installed
# !pip install transformers

import torch
from transformers import BertTokenizer, BertModel
import pyJoules  
import random
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

def initialize_bert_model():
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def encode_text(tokenizer, text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return inputs

def get_bert_embeddings(model, inputs):
    # Get BERT embeddings for input text
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    return embeddings

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Initialize BERT model and tokenizer
    tokenizer, model = initialize_bert_model()
    
    # Example text
    text = "Transformers are awesome!"

    # Encode text and get embeddings
    inputs = encode_text(tokenizer, text)
    embeddings = get_bert_embeddings(model, inputs)

    # Display embeddings shape
    #print("Embeddings shape:", embeddings.shape)


if __name__ == "__main__":
    main()
    
csv_handler.save_data()

