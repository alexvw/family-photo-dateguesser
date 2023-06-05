# family-photo-dateguesser

## A family photo processor - given known faces and birthdays, guess when a photo was taken

Expects the following file structure:

```
/people
  /person_name_1
    /birthyear.txt
    /photo_of_person1_only.jpg
  /person_name_2
    /birthyear.txt
    /photo_of_person2_only.bmp
    ...
  ...
/to_classify
  /photo_to_classify.jpeg
  
pretrained_model_goes_here
```

Folder "/people" for people, containing subfolder named after each person, containing training images for that person's face. Also include a birthyear.txt containing only the 4-digit birth year for this person)

Folder "/to_classify" containing images that have these people in them, that you want to classify.

Pretrained model not included (I dont think I am allowed to include this)

### How it works:
1. Preprocesses all training images, cropping to 120% of the face, and resizing. 
2. Uses face images to train a model, starting with a pretrained resnet50 and adding another layer with a dimension for each person it was trained on. Model will be named something like person1_person2_person3.model
3. Goes through every file in the to_classify folder, performing the following:
  a. Detect faces in the image
  b. For each face, attempt to identify the person
  c. Use this identity and the known birthday, to guess the date of the photo taken
  d. If there are more than one known faces, average the date taken
  e. If this date is reasonable, tag photo exif with date probably taken and people in image
  
gitignore is set up to hopefully prevent any PII from leaking to github, but be careful
  
### Does it actually work?
Not particularly, not yet. Lots of issues, particularly with the models. Tweaking model params to try and get something usable

### Credit to GPT4 for helping me troubleshoot the model params
