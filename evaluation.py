import argparse
import os
import sys
import warnings
import torch
import torch.nn.functional as F
import gc
torch.cuda.empty_cache()
gc.collect()

sys.path.append('C:\\Users\\max.polak\\PycharmProjects\\tactile-sensing\\master_students\\max_tactile_xai\\training')
sys.path.append('C:\\Users\\max.polak\\PycharmProjects\\tactile-sensing\\master_students\\max_tactile_xai\\RISE')
sys.path.append('C:\\Users\\max.polak\\PycharmProjects\\tactile-sensing\\master_students\\max_tactile_xai\\ShuffleNet-Series')
sys.path.append('C:\\Users\\max.polak\\PycharmProjects\\tactile-sensing\\master_students\\max_tactile_xai\\ShuffleNet-Series\\ShuffleNetV2+')
warnings.filterwarnings('ignore')

from model import LitModelRegression
from data_loader import ImageTransform
from config import CONFIG

from benchmarks.utils import get_data, TactileImageDatasetExplain
from benchmarks.metrics import Insertion, Deletion

import numpy as np

import pandas as pd

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Saliency map faithfullness metrics evaluation from npz")

#########################
#### data parameters ####
#########################
trial = 'trial77'
parser.add_argument("--data_folder", type=str, default=f"{os.path.expanduser('~')}/Data/datacollection_v3_november/explain/{trial}/train_dataset",
                    help="path to data repository")

parser.add_argument("--model", type=str, default='shufflenetv2',
                    help="model architecture: shufflenetv2 (default) or shufflenetv2+")

parser.add_argument("--input_type", type=str, default='raw',
                    help="model input: raw (default) or diff")

parser.add_argument("--saliency_npz", type=str, default='',
                    help="saliency file")

parser.add_argument("--cuda", dest="gpu", action='store_true',
                    help="use cuda")
parser.add_argument("--cpu", dest="gpu", action='store_false',
                    help="use cpu instead of cuda (default)")
parser.set_defaults(gpu=False)

parser.add_argument("--npz_folder", type=str, default="./npz",
                    help="Path to the folder where npz are stored")
parser.add_argument("--csv_folder", type=str, default="./csv",
                    help="Path to the folder to store the csv outputs")
parser.add_argument("--batch_size", type=int, default=1,
                    help="max batch size, default to 1")

def main():
    print('Evaluate')
    global args
    args = parser.parse_args()

    registered_models = 'C:/Users/max.polak/PycharmProjects/tactile-sensing/master_students/max_tactile_xai/registered-models/'

    # Model selection
    CONFIG.model_name = args.model
    target_layer = None
    if args.model == 'shufflenetv2':
        if args.input_type == 'raw':
            weights = registered_models + 'raw-shufflenetv2.ckpt'
        else:
            weights = registered_models + 'diff-shufflenetv2.ckpt'

        model = LitModelRegression.load_from_checkpoint(checkpoint_path=weights).model
        target_layers = ['conv1', 'stage2', 'stage3', 'stage4', 'conv5']

    elif args.model == 'shufflenetv2+':
        if args.input_type == 'raw':
            weights = registered_models + 'raw-shufflenetv2+.ckpt'
        else:
            weights = registered_models + 'diff-shufflenetv2+.ckpt'

        model = LitModelRegression.load_from_checkpoint(checkpoint_path=weights).model
        target_layers = ['first_conv', 'features', 'conv_last', 'globalpool', 'LastSE']

    else:
        print("model: " + args.model + " unknown, please check again")

    model.eval()
    # model_softmax = torch.nn.Sequential(model, torch.nn.Softmax(dim=-1))
    if args.gpu:
        model = model.cuda()
        # model_softmax.cuda()

    input_size = 224

    # get saliencies from file
    saliencies = np.load(args.npz_folder + "/" + args.saliency_npz)

    # Set metrics, use input size as step size
    deletion = Deletion(model=model, saliency_method=args.saliency_npz.split('-')[0], input_type=args.input_type, step=input_size, batch_size=args.batch_size)

    # Read images and labels
    image_path_list, labels = get_data(data_folder=args.data_folder, input_type=args.input_type)
    dataset = TactileImageDatasetExplain(img_path_list=np.array(image_path_list), labels=labels,
                                         input_type=args.input_type,
                                         transform=ImageTransform(is_train=False, img_size=224))

    ins_auc_dict = dict()
    del_auc_dict = dict()
    ins_details_dict = dict()
    del_details_dict = dict()

    for i in tqdm(range(len(dataset))):

        # uncomment when creating masks
        if not i==6:
            continue

        # import image from dataset
        original_image, image, label, image_name = dataset[i]
        image = image.unsqueeze(0)

        # Load saliency
        saliency = torch.tensor(saliencies[image_name])
        # upscale saliency
        sh, sw = saliency.shape[-2:]
        saliency = saliency.view(1,1,sh,sw)
        saliency = F.interpolate(saliency, image.shape[-2:], mode='bilinear')

        # set image and saliency to gpu if required
        if args.gpu:
            image = image.cuda()
            saliency = saliency.cuda()

        # get output predicted by the model for the full image, it's the class used to generate saliency map
        # model_pred = model(image)
        # score = torch.clip(torch.round(model_pred), 0, 100).item()
        # print(score)

        # compute insertion and deletion for each step + auc on the image
        del_auc, del_details = deletion(image, saliency, class_idx=None)
        # print(del_auc)

        # store every values for the image in dictionary
        del_auc_dict[image_name] = del_auc.cpu().numpy()
        del_details_dict[image_name] = del_details.cpu().numpy()


    csv_suffix = '.'.join(args.saliency_npz.split('.')[:-1]) + ".csv"

    # save scores in csv files
    pd.DataFrame.from_dict(del_auc_dict, orient='index').to_csv(args.csv_folder + "/" + 'del_auc_' + csv_suffix)
    pd.DataFrame.from_dict(del_details_dict, orient='index').to_csv(args.csv_folder + "/" + 'del_details_' + csv_suffix)

if __name__ == "__main__":
    main()
