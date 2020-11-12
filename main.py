from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models



def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image is <= 400 pixels in the x-y dims.'''
    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image



# helper function for un-normalizing an image and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def imshow(img):              # Pour afficher une image
    plt.figure(1)
    plt.imshow(img)
    plt.show()


### Run an image forward through a model and get the features for a set of layers. 'model' is supposed to be vgg19
def get_features(image, model, layers=None):  
    if layers is None:
        layers = {'0': 'conv0',
                  '5': 'conv5', 
                  '10': 'conv10', 
                  '19': 'conv19'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
   # tensor: Nfeatures x H x W ==> M = Nfeatures x Npixels with Npixel=HxW
   ...
   return gram

if __name__ == '__main__':

    print(torch.cuda.device_count() )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)

    ########################## DISPLAY IMAGE#########################################################""
    content = load_image('images/montagne.jpg').to(device)
    style = load_image('images/peinture1.jpg', shape=content.shape[-2:]).to(device)

    imshow(im_convert(content))
    imshow(im_convert(style))


    vgg = models.vgg19(pretrained=True).features

    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)

    features = list(vgg)[:23]
    for i,layer in enumerate(features):
        print(i,"   ",layer)

    target = content.clone().requires_grad_(True).to(device)

    optimizer = optim.Adam([target], lr=0.003)
    for i in range(2000):
    
        # get the features from your target image
        features_content = get_features(content,vgg,None)
        features_style = get_features(style,vgg,None)
        features_target = get_features(target,vgg,None)
        print(i)

    
        # the content loss
        loss_content= torch.mean((features_content["conv19"]-features_target["conv19"])**2)
 
        
        # the style loss
        loss_style = torch.mean((features_style["conv10"]-features_target["conv10"])**2) + torch.mean((features_style["conv5"]-features_target["conv5"])**2) + torch.mean((features_style["conv0"]-features_target["conv0"])**2)
        
        # calculate the *total* loss
        total_loss=0.80*loss_style + 0.2*loss_content
    
        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    imshow(im_convert(target))

