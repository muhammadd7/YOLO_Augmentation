# YOLO_Augmentation
To apply different augmentations, you can simply add these into A.compose() in augalbem.py

# define the augmentations to be applied

transform = A.Compose([

A.Flip(), # flip images horizontally

A.Rotate(limit=45), # rotate images by a random degree between -45 and 45

A.RandomBrightnessContrast(), # change brightness and contrast of images

A.RandomGamma(), # change gamma of images

A.GaussianBlur(), # blur images using Gaussian blur

A.GaussNoise(), # add Gaussian noise to images

A.Superpixels (p_replace=0.1, n_segments=100, max_size=128, interpolation=1, always_apply=False, p=0.5), # randomly drop out pixels in images

A.RandomScale(), # randomly scale images

A.HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)

], bbox_params=A.BboxParams(

format='yolo', label_fields=['class_labels']

))
