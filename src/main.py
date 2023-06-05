# main.py
from preprocess import preprocess_images
from train import train_model
from classify import classify_face

def main():
    # Preprocess images
    print("Preprocessing images...")
    preprocess_images()

    # Train model
    print("Training model...")
    model_path, class_to_idx = train_model()

    # Classify images
    print("Classifying images...")
    classify_face(model_path, class_to_idx)

if __name__ == "__main__":
    main()