import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from u2net.data_loader import RescaleT
from u2net.data_loader import ToTensor
from u2net.data_loader import ToTensorLab
from u2net.data_loader import SalObjDataset

from u2net.model import U2NET # full size version 173.6 MB
from u2net.model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def save_output(image_path, pred, prediction_dir, root_dir):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    # Convert normalized prediction to an RGB image
    imo = Image.fromarray((predict_np * 255).astype(np.uint8)).convert('RGB')

    # Read original image to get its width/height
    image = io.imread(image_path)
    imo = imo.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    # Derive the output filename
    img_name = os.path.basename(image_path)
    # base_name, _ = os.path.splitext(img_name)
    out_filename = img_name

    # Get the subdirectory structure relative to the root_dir
    rel_path = os.path.relpath(image_path, root_dir)  # e.g. "dog_images/labrador/img001.jpg"
    rel_folder = os.path.dirname(rel_path)            # e.g. "dog_images/labrador"

    # Create the corresponding folder under prediction_dir
    out_dir = os.path.join(prediction_dir, rel_folder)
    os.makedirs(out_dir, exist_ok=True)

    # Full path to where we'll save the prediction
    out_path = os.path.join(out_dir, out_filename)
    imo.save(out_path)

def get_image_paths(root_dir, exts=('.png','.jpg','.jpeg','.bmp','.tiff','.tif')):
    """
    Recursively gather image file paths from `root_dir`,
    returning only files with extensions in `exts`.
    """
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # Check file extension
            if filename.lower().endswith(exts):
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)
    return image_paths

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'
    # model_name='u2netp'

    image_dir = os.path.join(os.getcwd(), 'datasets', 'PACS')
    prediction_dir = os.path.join(os.getcwd(), 'datasets', 'PACS', 'masks')
    model_dir = os.path.join(os.getcwd(),'u2net', 'saved_models', model_name, model_name + '.pth')

    img_name_list = get_image_paths(image_dir)
    print("Found image paths:", img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=4)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(
                img_name_list[i_test],   # original image path
                pred,
                prediction_dir,
                image_dir               # the “root” of your images
            )
        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
