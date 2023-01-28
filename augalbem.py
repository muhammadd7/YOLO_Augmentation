import albumentations as A
import os
import cv2

# # directory containing original images
# image_dir = "E:/CUI data/FYP Work/Datasets/Dataset(All 7 Classes)/fight/images"
# # directory containing original bounding box labels
# label_dir = "E:/CUI data/FYP Work/Datasets/Dataset(All 7 Classes)/fight/labels"

# directory containing original images
image_dir = "C:/Users/HP/Desktop/trainy/images"
# directory containing original bounding box labels
label_dir = "C:/Users/HP/Desktop/trainy/labels"
# directory to save augmented images
save_dir = "C:/Users/HP/Desktop/trainy/Aimages"
# directory to save augmented bounding box labels
save_label_dir = "C:/Users/HP/Desktop/trainy/Alabels"


# define the augmentations to be applied
transform = A.Compose([
    A.Flip(), # flip images horizontally
    A.Rotate(limit=45), # rotate images by a random degree between -45 and 45
    A.RandomBrightnessContrast() # change brightness and contrast of images
], bbox_params=A.BboxParams(
            format='yolo', label_fields=['class_labels']
        ))

# loop through all images in the original image directory
for filename in os.listdir(image_dir):
    # check if file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # read image and bounding boxes
        image = cv2.imread(os.path.join(image_dir, filename))
        
        label_file = os.path.splitext(filename)[0] + '.txt'
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
        bboxes = []
        labels = []
        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            labels.append(class_id)
            x1, y1, x2, y2 = map(float, data[1:])
            bboxes.append([x1, y1, x2, y2])
            
        # convert bounding boxes to albumentations format
        bboxes_albu = [A.BboxParams(bbox) for bbox in bboxes]
        for i in range(5):
            augmented = transform(
                image=image, bboxes=bboxes, class_labels=labels
            )

            image_aug = augmented['image']
            bboxes_aug = augmented['bboxes']
            labels_aug = augmented['class_labels']
            # write augmented image to save directory
            # cv2.imwrite(os.path.join(save_dir, filename), image_aug)
            # # convert augmented bounding boxes back to yolo format
            # with open(os.path.join(save_label_dir, label_file), 'w') as f:
            #     for idx in range(len(bboxes_aug)):
            #         x1, y1, x2, y2 = bboxes_aug[idx]
            #         class_id = labels_aug[idx]
            #         f.write(f"{class_id} {x1} {y1} {x2} {y2}\n")
            
            # write augmented image to save directory
            new_filename = f"{os.path.splitext(filename)[0]}_{i}.{os.path.splitext(filename)[1]}"
            cv2.imwrite(os.path.join(save_dir, new_filename), image_aug)
            # convert augmented bounding boxes back to yolo format
            new_label_file = os.path.splitext(new_filename)[0] + '.txt'
            with open(os.path.join(save_label_dir, new_label_file), 'w') as f:
                for idx in range(len(bboxes_aug)):
                    x1, y1, x2, y2 = bboxes_aug[idx]
                    class_id = labels_aug[idx]
                    f.write(f"{class_id} {x1} {y1} {x2} {y2}\n")
