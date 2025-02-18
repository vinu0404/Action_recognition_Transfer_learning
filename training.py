from data_preprocessing import load_and_process_data
from model_definition import create_model
from config import EPOCHS, BATCH_SIZE, MODEL_PATH

def train_model():
    features, targets = load_and_process_data()
    model = create_model()
    
    history = model.fit(features, targets, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_split=0.2)
    
    model.save(MODEL_PATH)
    return model, history

if __name__ == "__main__":
    model, history = train_model()
    print("Training completed. Model saved to", MODEL_PATH)