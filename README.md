# YOLO_Augmentation
To apply different augmentations, you can simply add these into A.compose() in augalbem.py

transform = A.Compose([

A.Flip(), # flip images horizontally

A.Rotate(limit=45), # rotate images by a random degree between -45 and 45

A.RandomBrightnessContrast(), # change brightness and contrast of images

A.RandomGamma(), # change gamma of images

A.GaussianBlur(), # blur images using Gaussian blur

A.GaussNoise(), # add Gaussian noise to images

A.CoarseDropout(), # randomly drop out pixels in images

A.RandomScale(), # randomly scale images

A.RandomCrop(), # randomly crop images

A.RandomBrightness(), # randomly change brightness of images

A.RandomHueSaturationValue(), # randomly change hue, saturation and value of images

A.Cutout() # cut out a random rectangular section of an image

], bbox_params=A.BboxParams(

format='yolo', label_fields=['class_labels']

))
