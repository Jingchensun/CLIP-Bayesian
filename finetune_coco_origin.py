#https://github.com/openai/CLIP/issues/57

import os
import torch
import glob
from PIL import Image
import random
import clip
from tqdm.notebook import tqdm
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm

EPOCH =10
BATCH_SIZE =64

device = "cuda:2" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

class cocodtrain(torch.utils.data.Dataset):
    def __init__(self, image_path='/home/jason/data/coco2014/images', text_path='/home/jason/data/coco2014/text', mode='train2014'):

        self.image_list = []
        self.image_list.extend(glob.glob(os.path.join(image_path, mode, '*.jpg')))
        self.image_list.sort()

        self.label_list = []
        self.label_list.extend(glob.glob(os.path.join(text_path, mode, '*.txt')))
        self.label_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        image = image.resize((224,224), Image.BILINEAR)
        image = preprocess(image)
        #image = np.asarray(image)

        with open(self.label_list[index], "r") as f:
            data = f.readlines()
            label = random.choice(data)
            
        return image, label
trainset = cocodtrain('/home/jason/data/coco2014/images','/home/jason/data/coco2014/text','train2014')
trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=BATCH_SIZE,
                    shuffle=True, 
                    num_workers=16,
                    drop_last=True)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

#device = "cuda:3" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
#model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

#clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in range(EPOCH):
    print('epoch:', epoch)
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        list_image,list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
        # print(list_image.size()) #torch.Size([32, 3, 224, 224])
        print(len(list_txt))
      
        images = torch.tensor(np.stack(list_image)).to(device)
        texts = clip.tokenize(list_txt).to(device) #torch.Size([32, 77])
         # print(texts.size()) #torch.Size([32, 77])
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(BATCH_SIZE,dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()
        print('total loss:', total_loss)
      
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)
    
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': total_loss,
    }, f"model_checkpoint/model_10.pt") #just change to your preferred folder/filename    