import torch
import torch.nn as nn
import spacy
import json
import os
from .config import (
    EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, 
    DROPOUT, MAX_LENGTH, LABEL_MAPPING
)

class BiLSTM(nn.Module):
    """
    BiLSTM Architecture as defined in the emotion_classifier.ipynb
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 dropout, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # Bi-LSTM Layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        
        final_hidden = self.dropout(torch.cat((hidden_forward, hidden_backward), dim=1))
        return self.fc(final_hidden)


class EmotionClassifier:
    """
    Inference class to classify emotions based on the pre-trained BiLSTM.
    """
    def __init__(self, model_path: str, vocab_path: str = None, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # We need a vocabulary dictionary. If passed, load it.
        # Ensure to save your vocab.json during training to be used here.
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                self.vocab_dict = json.load(f)
        else:
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please provide a valid vocab.json.")
            
        vocab_size = len(self.vocab_dict)
        
        # Load spaCy for tokenization
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner', 'lemmatizer'])
        except OSError:
            # Fallback message just in case the space model is not downloaded
            import warnings
            warnings.warn("en_core_web_sm not found. Please run 'python -m spacy download en_core_web_sm'")
            self.nlp = spacy.blank("en")
            
        # Init model
        self.model = BiLSTM(
            vocab_size=max(vocab_size, 2), # At least 2 for pad and unk
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            dropout=DROPOUT
        ).to(self.device)
        
        # Load weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model .pth file not found at {model_path}. Please provide a valid trained model.")
        
        self.model.eval()

    def tokenizer(self, text: str):
        return [token.text.lower() for token in self.nlp(text) if not token.is_punct and not token.is_space]

    def text_to_indices(self, text: str):
        tokens = self.tokenizer(text)
        return [self.vocab_dict.get(token, 1) for token in tokens] # 1 is index for <unk>

    def predict_emotion(self, question: str) -> dict:
        """Main method to predict the emotion and return a structured response."""
        indices = self.text_to_indices(question)
        
        # Truncate
        if len(indices) > MAX_LENGTH:
            indices = indices[:MAX_LENGTH]
            
        tensor = torch.tensor(indices, dtype=torch.int64).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            prediction = torch.argmax(outputs, dim=1).item()
            
        emotion = LABEL_MAPPING.get(prediction, "unknown")
        
        return {
            "text": question,
            "emotion": emotion
        }
