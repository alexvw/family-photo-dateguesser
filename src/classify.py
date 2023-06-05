# classify.py
import os
import torch
from PIL import Image
import io, piexif
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from age_estimation import estimate_age

def classify_face(model_path, class_to_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = 'people'  # Directory where training data is located

    # Invert the dictionary
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Load the saved model
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # Prepare the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Prepare the dataset and dataloader
    dataset_dir = 'to_classify'  # Directory where unclassified images are stored
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(dataset_dir, filename)
            img = Image.open(image_path)

            # Detect the face
            box, _ = mtcnn.detect(img)
            if box is not None:
                box = box[0]  # We only take the first detected face

                # Crop the face out of the image
                face = img.crop(box)
                face_tensor = transform(face)
                face_tensor = face_tensor.unsqueeze(0).to(device)

                # Classify the face
                outputs = model(face_tensor)
                _, predicted = torch.max(outputs, 1)
                person_index = predicted.item()

                
                # Get the person's name from the predicted index
                person_name = idx_to_class[person_index]

                # Load birth year
                birth_year_file = os.path.join('people', person_name, 'birthyear.txt')
                with open(birth_year_file, 'r') as f:
                    birth_year = int(f.read().strip())

                # Estimate age
                estimated_age = estimate_age(face)

                # Estimate photo year
                photo_year = birth_year + estimated_age

                # Open image
                image = Image.open(image_path)

                if "exif" in image.info:
                    exif_dict = piexif.load(image.info["exif"])
                else:
                    exif_dict = piexif.makeExif()  # Create empty exif data if not present

                # Add photo year as metadata to ImageDescription tag
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = str(photo_year)

                # Dump the exif data back into the image and save it
                exif_bytes = piexif.dump(exif_dict)
                image.save(image_path, exif=exif_bytes)

                print(f'Image {filename} is classified as person {person_index} and estimated photo year is {photo_year}.')


if __name__ == "__main__":
    classify_face()