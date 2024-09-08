import os
import torch
from PIL import Image, ImageTk
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import tkinter as tk
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.models import resnet18

# Classe del Dataset
class ChallengeTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = os.listdir(data_dir)
        self.transform = transform
        self.used_images = []  # Lista per tenere traccia delle immagini già usate

    def __len__(self):
        return len(self.image_paths) - len(self.used_images)  

    def __getitem__(self, idx):
        remaining_images = list(set(self.image_paths) - set(self.used_images))
        image_path = os.path.join(self.data_dir, remaining_images[idx])
        self.used_images.append(remaining_images[idx]) 
        
        image = Image.open(image_path).convert('RGB')  

        if self.transform:
            image = self.transform(image)
        return image, image_path

# Funzione principale
def conferma():
    global img_tk  
    global img_tk2
    
    if len(test_dataset) == 0:  # Se non ci sono più immagini disponibili
            result_label.config(text="Tutte le immagini sono state elaborate!")
            return
        
    random_index = np.random.randint(0, len(test_dataset))
    img, path = test_dataset[random_index]

    original_path = path.replace('new_dataset', 'dataset')
    real_img = Image.open(original_path)

    # Mostra real_img in canvas
    img_tk2 = ImageTk.PhotoImage(real_img)
    canvas.create_image(0, 0, anchor="nw", image=img_tk2)
    canvas.image = img_tk2

    # Invertire la normalizzazione
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_inv = inv_normalize(img)
    img_inv = TF.to_pil_image(img_inv)
    img_tk = ImageTk.PhotoImage(img_inv)

    canvas2.create_image(0, 0, anchor="nw", image=img_tk)
    canvas2.image = img_tk  

    input_img = img.unsqueeze(0)
    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[pred_label_idx.item()]

    description_label.config(text="Immagine preprocessata:")
    result_label.config(text="Predicted: {} ({})".format(predicted_label, prediction_score.squeeze().item()))

# Caricamento del dataset e modello
print("Caricamento dataset...")
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256 pixels
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# INSERIRE PERCORSO DEL DATASET PREPROCESSATO
test_dataset = ChallengeTestDataset(
    data_dir='C:/Users/alexn/Desktop/demo/datasets/new_dataset/test',
    transform=transform
)
print("Dataset caricato")
print("Proseguire nella GUI")
idx_to_labels = {0: 'plug adapter', 1: 'telephone', 2: 'scissor', 3: 'light bulb', 4: 'pepsi can', 5: 'sun glasses', 6: 'mini ball', 7: 'cup', 8: 'book'}

# Caricamento del modello
model = resnet18(weights=None)
num_classes = 9
model.fc = nn.Linear(model.fc.in_features, num_classes)

# INSERIRE PERCORSO PESI MODELLO
model_weights_path = "C:/Users/alexn/Desktop/demo/weights/weights_and_lr.pth"
checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Creazione della GUI
window = tk.Tk()
window.title("Menu")

# Centra la finestra principale
window_width = 500
window_height = 800
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Testo introduttivo
intro_label = tk.Label(window, text="Benvenuti nella Demo di 'Tackling Domain Shift using Fast R-CNN and ResNet18' \n " +
                       "Clicca per procedere:", wraplength=500, pady=5)
intro_label.pack()

# Bottone di conferma
confirm_button = tk.Button(window, text="Elabora", command=conferma)
confirm_button.pack()

# Canvas per disegnare l'immagine originale
canvas = tk.Canvas(window, width=350, height=350)
canvas.pack()

# Testo introduttivo
description_label = tk.Label(window, text="", wraplength=500, pady=5)
description_label.pack()

# Canvas per disegnare la test image preprocessata
canvas2 = tk.Canvas(window, width=256, height=256)
canvas2.pack()

# Per il risultato
result_label = tk.Label(window, text="")
result_label.pack()

window.mainloop()
