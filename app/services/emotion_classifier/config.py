# Model Hyperparameters based on the best weighted glove model
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 6
N_LAYERS = 2
DROPOUT = 0.5
MAX_LENGTH = 53

# Label mapping from notebook
LABEL_MAPPING = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
