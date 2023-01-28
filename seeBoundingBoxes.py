import os
import cv2

# Define the directories for images and labels
# image_dir = "C:/Users/HP/Desktop/train/images"
# label_dir = "C:/Users/HP/Desktop/train/labels"

# directory to save augmented images
image_dir = "C:/Users/HP/Desktop/trainy/Aimages"
# directory to save augmented bounding box labels
label_dir = "C:/Users/HP/Desktop/trainy/Alabels"

def yolo_to_x_y(x_center, y_center, x_width, y_height, width, height):
    x1 = int((x_center - x_width/2) * width)
    y1 = int((y_center - y_height/2) * height)
    x2 = int((x_center + x_width/2) * width)
    y2 = int((y_center + y_height/2) * height)
    return x1, y1, x2, y2

# Loop through all images in the image directory
for image_name in os.listdir(image_dir):
    # Load the image
    img = cv2.imread(os.path.join(image_dir, image_name))
    # Get the corresponding label file path
    label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
    # Check if the label file exists
    if os.path.isfile(label_path):
        with open(label_path) as f:
            content = f.readlines()
        # Loop through all lines in the label file
        for line in content:
            values_str = line.split()
            class_index, x_center, y_center, x_width, y_height = map(float, values_str)
            class_index = int(class_index)
            # Convert yolo to points
            x1, y1, x2, y2 = yolo_to_x_y(x_center, y_center, x_width, y_height, img.shape[1], img.shape[0])
            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            # Display the image with bounding boxes
            cv2.imshow('Image with bounding boxes', img)
            cv2.waitKey(0)
    else:
        print(f"Label file for {image_name} not found")
cv2.destroyAllWindows()
