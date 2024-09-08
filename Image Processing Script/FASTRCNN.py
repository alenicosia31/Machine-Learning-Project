import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import random
from matplotlib.patches import Rectangle

def visualize_image(title, image, cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_detections(image, boxes, box_ids):
    fig, ax = plt.subplots()
    ax.imshow(image)
    colors = plt.cm.get_cmap('tab10', len(boxes))  # Generate a colormap with a different color for each box
    
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        color = colors(i)  # Get a color from the colormap
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 10, f'ID: {box_ids[i]}', color=color, fontsize=12, weight='bold')
    
    plt.axis('off')
    chosen_idx = [None]
    
    def on_key(event):
        if event.key.isdigit():
            idx = int(event.key)
            if idx in box_ids:
                chosen_idx[0] = idx
                plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    
    return chosen_idx[0]

def crop_image(image_path, bbox):
    # Read the image
    image = cv2.imread(image_path)
    
    # Extract the bounding box coordinates
    x1, y1, x2, y2 = bbox
    
    cropped_image = image[y1:y2, x1:x2]
    
    # Convert the cropped image to RGB (from BGR)
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    return cropped_image_rgb

def overlay_object_on_background(object_image):
    background_path = r"C:\Users\alexn\Desktop\IMG.jpeg"
    # Read the background image and resize it
    background = cv2.imread(background_path)
    background_resized = cv2.resize(background, (350, 350))
    background_resized_rgb = cv2.cvtColor(background_resized, cv2.COLOR_BGR2RGB)
    
    
    obj_height, obj_width = object_image.shape[:2]
    
    # Create a random position to place the resized object
    max_x = background_resized_rgb.shape[1] - obj_width
    max_y = background_resized_rgb.shape[0] - obj_height
    random_x = random.randint(20, max_x)
    random_y = random.randint(20, max_y)
    
    # Overlay the resized object onto the background at the random position
    for c in range(0, 3): # Iterate over the color channels and for each of them selecte the region of interest on the background image 
        background_resized_rgb[random_y:random_y+obj_height, random_x:random_x+obj_width, c] = np.where( 
            object_image[:, :, c] == 0, # Check if the obj pixel is black
            background_resized_rgb[random_y:random_y+obj_height, random_x:random_x+obj_width, c], # if black, the pixel is sub with the original pixel of the background
            object_image[:, :, c] # if not black, the pixel from the obj image is used
        )
    
    return background_resized_rgb

def get_model():
    # resnet50 as feature extractor
    # feature pyramid network helps fast r cnn to detect small objects
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(image_path, model):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    return predictions[0], image

def distance(box1, box2):
    # Calculate distance between centers of two bounding boxes
    center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
    center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
    return np.linalg.norm(center1 - center2)

def merge_close_boxes(boxes, distance_threshold=20):
    merged_boxes = []
    merged_flags = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if merged_flags[i]:
            continue
        
        merged_box = boxes[i]
        count = 1
        
        for j in range(i + 1, len(boxes)):
            if merged_flags[j]:
                continue
            
            dist = distance(boxes[i], boxes[j])
            
            if dist < distance_threshold:
                merged_box = [
                    min(merged_box[0], boxes[j][0]),
                    min(merged_box[1], boxes[j][1]),
                    max(merged_box[2], boxes[j][2]),
                    max(merged_box[3], boxes[j][3])
                ]
                merged_flags[j] = True
                count += 1
        
        merged_boxes.append(merged_box)
        merged_flags[i] = True
    
    return merged_boxes

def main():
    model = get_model()
    dir_path = r"C:\Users\alexn\Desktop\dataset"
    save_path = r"C:\Users\alexn\Desktop\new_dataset"
    test_path = os.path.join(dir_path, "test")
    
    counter = 0
    print("Test images processed:")
    
    image_files = sorted([f for f in os.listdir(test_path) if f.endswith('.jpg')])
    for image_file in image_files:
        for i in range(0, 800):
            if image_file != f"{i:03}.jpg":
                continue
            image_path = os.path.join(test_path, image_file)
            
            predictions, image = detect_objects(image_path, model)
            
            boxes = []
            for idx, score in enumerate(predictions['scores']):
                if score > 0.5:  # confidence threshold
                    bbox = predictions['boxes'][idx].numpy().astype(int)
                    boxes.append(bbox)
                    
            boxes = merge_close_boxes(boxes, 30)
            if boxes:
                if len(boxes) > 2:
                    print("OKAY")
                    # Visualize detections and get user choice
                    print(len(boxes))
                    
                    chosen_idx = visualize_detections(image, boxes, list(range(len(boxes))))
                    
                else:
                    # Select the smallest bounding box by area
                    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                    chosen_idx = areas.index(min(areas))
                
                if chosen_idx is not None: 
                    selected_box = boxes[chosen_idx]
                    
                    object_image = crop_image(image_path, selected_box)
                    result_image = overlay_object_on_background(object_image)
                    
                    save_folder = os.path.join(save_path, "test")
                    plt.imsave(os.path.join(save_folder, image_file), result_image)
                    
                    counter += 1
                    print(counter)

if __name__ == "__main__":
    main()