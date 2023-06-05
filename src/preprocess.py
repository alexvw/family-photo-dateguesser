# preprocess.py
import os
from PIL import Image
from facenet_pytorch import MTCNN

def preprocess_images():
    # Initialize MTCNN
    mtcnn = MTCNN()

    data_dir = 'people'  # Directory where training data is located

    # Get the names of people
    people_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    for name in people_names:
        print(f'Processing images of {name}')
        person_dir = os.path.join(data_dir, name)
        for filename in os.listdir(person_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Add here any file types you need
                image_path = os.path.join(person_dir, filename)
                img = Image.open(image_path)

                # Detect faces in the image
                boxes, _ = mtcnn.detect(img)

                if boxes is not None:
                    # Assume the first box is the face of the person of interest
                    box = boxes[0]

                    # Add a margin around the face
                    width, height = box[2] - box[0], box[3] - box[1]
                    margin = [width * 0.2, height * 0.2]  # 20% margin
                    box[0] = max(0, box[0] - margin[0])
                    box[1] = max(0, box[1] - margin[1])
                    box[2] = min(img.width, box[2] + margin[0])
                    box[3] = min(img.height, box[3] + margin[1])

                    # Crop the face from the image and save
                    face = img.crop(box)
                    face.save(image_path)

if __name__ == "__main__":
    preprocess_images()