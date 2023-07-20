import argparse
import os
import sys
import torch
import warnings
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.append('C:\\Users\\max.polak\\PycharmProjects\\tactile-sensing\\master_students\\max_tactile_xai\\training')
sys.path.append('C:\\Users\\max.polak\\PycharmProjects\\tactile-sensing\\master_students\\max_tactile_xai\\RISE')
sys.path.append('C:\\Users\\max.polak\\PycharmProjects\\tactile-sensing\\master_students\\max_tactile_xai\\ShuffleNet-Series')
sys.path.append('C:\\Users\\max.polak\\PycharmProjects\\tactile-sensing\\master_students\\max_tactile_xai\\ShuffleNet-Series\\ShuffleNetV2+')
warnings.filterwarnings('ignore')

from polycam.polycam import PCAMp, PCAMm, PCAMpm
from benchmarks.utils import get_data, TactileImageDatasetExplain
from benchmarks.utils import overlay
import time

from model import LitModelRegression
from data_loader import ImageTransform
from config import CONFIG

try:
    from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM, LayerCAM
    torchcam = True
except:
    print("torchcam not installed: GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM not availables")
    torchcam = False

try:
    from explanations import RISE
    rise = True
except:
    print("RISE not installed, not available")
    rise = False

try:
    from captum.attr import IntegratedGradients, InputXGradient, Lime, Occlusion, Saliency, NoiseTunnel, GuidedBackprop, DeepLift, DeepLiftShap
    from captum.metrics import sensitivity_max, infidelity
    captum = True
except:
    print("captum not installed, IntegratedGradients, InputXGradient, SmoothGrad, Occlusion not availables")
    captum = False

import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Saliency map methods evaluation")

#########################
#### data parameters ####
#########################
trial = 'trial77' # trial77, trial83, complete-test

parser.add_argument("--data_folder", type=str, default=f"{os.path.expanduser('~')}/Data/datacollection_v3_november/explain/{trial}/train_dataset",
                    help="path to data repository")

parser.add_argument("--model", type=str, default='shufflenetv2',
                    help="model architecture: shufflenetv2 (default) or shufflenetv2+")

parser.add_argument("--input_type", type=str, default='raw',
                    help="model input: raw (default) or diff")

parser.add_argument("--saliency", type=str, default='pcampm',
                    help="saliency type: pcamp, pcamm, pcampm (default), gradcam, gradcampp, smoothgradcampp, "
                         "bg (=GuidedBackprop), ig (=IntegratedGrad), ixg (=InputxGrad), sg (=SmoothGrad), dl (=DeepLift), dlshap (=DeepLiftShap), occlusion, rise")

parser.add_argument("--cuda", dest="gpu", action='store_true',
                    help="use cuda")
parser.add_argument("--cpu", dest="gpu", action='store_false',
                    help="use cpu instead of cuda (default)")
parser.set_defaults(gpu=False)

parser.add_argument("--batch_size", type=int, default=1,
                    help="max batch size (when saliency method use it), default to 1")

parser.add_argument("--save_folder", type=str, default=f"./explanations/{trial}",
                    help="Path to the folder to store the output file")
parser.add_argument("--suffix", type=str, default="",
                    help="Add SUFFIX string to the checkpoint name")

def main():
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

    if args.saliency.lower() == 'rise':
        if not rise:
            print("Cannot use rise, import not available")
            return
        model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
        for p in model.parameters():
            p.requires_grad = False

    if args.saliency.lower() in ["ig", "ixg", "occlusion", "lime", "sg"] and not captum:
        print("cannot use captum methods, import not available")
        return

    if args.saliency.lower() in ["gradcam", "scorecam", "gradcampp", "smoothgradcampp", "sscam", "iscam"] and not torchcam:
        print("cannot use torchcam methods, import not available")
        return

    model.eval()
    # set model to cuda if required
    if args.gpu:
        model = model.cuda()

    input_size = 224

    # Saliency selection
    n_maps = 1
    library = None
    if args.saliency.lower() == 'pcamp':
        saliency = PCAMp(model, batch_size=args.batch_size)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'pcamm':
        saliency = PCAMm(model, batch_size=args.batch_size)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'pcampm':
        saliency = PCAMpm(model, batch_size=args.batch_size)
        n_maps = 5
        library = "polycam"
    if args.saliency.lower() == 'pcampinterm':
        saliency = PCAMp(model, batch_size=args.batch_size, target_layer_list=target_layers, intermediate_maps=True)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'pcamminterm':
        saliency = PCAMm(model, batch_size=args.batch_size, target_layer_list=target_layers, intermediate_maps=True)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'pcampminterm':
        saliency = PCAMpm(model, batch_size=args.batch_size, target_layer_list=target_layers, intermediate_maps=True)
        n_maps = 5
        library = "polycam"
    if args.saliency.lower() == 'pcampnolnorm':
        saliency = PCAMp(model, batch_size=args.batch_size, target_layer_list=target_layers, lnorm=False)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'pcammnolnorm':
        saliency = PCAMm(model, batch_size=args.batch_size, target_layer_list=target_layers, lnorm=False)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'pcampmnolnorm':
        saliency = PCAMpm(model, batch_size=args.batch_size, target_layer_list=target_layers, lnorm=False)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'gradcam':
        saliency = GradCAM(model, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'gradcam1':
        saliency = GradCAM(model, target_layer=target_layers[-2])
        library = "torchcam"
    elif args.saliency.lower() == 'gradcam2':
        saliency = GradCAM(model, target_layer=target_layers[-3])
        library = "torchcam"
    elif args.saliency.lower() == 'gradcam3':
        saliency = GradCAM(model, target_layer=target_layers[-4])
        library = "torchcam"
    elif args.saliency.lower() == 'gradcam4':
        saliency = GradCAM(model, target_layer=target_layers[-5])
        library = "torchcam"
    elif args.saliency.lower() == 'gradcampp':
        saliency = GradCAMpp(model, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'smoothgradcampp':
        saliency = SmoothGradCAMpp(model, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam':
        saliency = ScoreCAM(model, batch_size=args.batch_size, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam1':
        saliency = ScoreCAM(model, target_layer=target_layers[-2])
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam2':
        saliency = ScoreCAM(model, target_layer=target_layers[-3])
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam3':
        saliency = ScoreCAM(model, target_layer=target_layers[-4])
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam4':
        saliency = ScoreCAM(model, target_layer=target_layers[-5])
        library = "torchcam"
    elif args.saliency.lower() == 'sscam':
        saliency = SSCAM(model, batch_size=args.batch_size, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'iscam':
        saliency = ISCAM(model, batch_size=args.batch_size, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'layercam0':
        saliency = LayerCAM(model, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'layercam1':
        saliency = LayerCAM(model, target_layer=target_layers[-2])
        library = "torchcam"
    elif args.saliency.lower() == 'layercam2':
        saliency = LayerCAM(model, target_layer=target_layers[-3])
        library = "torchcam"
    elif args.saliency.lower() == 'layercam3':
        saliency = LayerCAM(model, target_layer=target_layers[-4])
        library = "torchcam"
    elif args.saliency.lower() == 'layercam4':
        saliency = LayerCAM(model, target_layer=target_layers[-5])
        library = "torchcam"
    elif args.saliency.lower() == 'rise':
        saliency = RISE(model, (input_size, input_size), args.batch_size)
        saliency.generate_masks(N=6000, s=8, p1=0.1)
        library = "rise"
    elif args.saliency.lower() == 'ig':
        saliency = IntegratedGradients(model)
        library = "captum"
    elif args.saliency.lower() == 'ixg':
        saliency = InputXGradient(model)
        library = "captum"
    elif args.saliency.lower() == 'lime':
        saliency = Lime(model)
        library = "captum"
    elif args.saliency.lower() == 'gb':
        saliency = GuidedBackprop(model)
        library = "captum"
    elif args.saliency.lower() == 'dl':
        saliency = DeepLift(model)
        library = "captum"
    elif args.saliency.lower() == 'dlshap':
        saliency = DeepLiftShap(model)
        library = "captum"
    elif args.saliency.lower() == 'sg':
        gradient = Saliency(model)
        sg = NoiseTunnel(gradient)
        def sg_fn(inputs, class_idx=0):
            return sg.attribute(inputs, nt_samples=50, nt_samples_batch_size=args.batch_size, target=class_idx).sum(1)
        saliency = sg_fn
        library = "overlay"
    elif args.saliency.lower() == 'occlusion':
        occlusion = Occlusion(model)
        def occ_fn(inputs, class_idx=0):
            return occlusion.attribute(inputs, target=class_idx, sliding_window_shapes=(3,64,64), strides=(3,8,8)).sum(1)
        saliency = occ_fn
        library = "overlay"
    # define a perturbation function for the input for explainability metric
    def perturb_fn(inputs):
        noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float().cuda()
        if args.gpu:
            noise = noise.cuda()
        return noise, inputs - noise

    # Initiate dataset
    # Read images and labels
    image_path_list, labels = get_data(data_folder=args.data_folder, input_type=args.input_type)

    dataset = TactileImageDatasetExplain(img_path_list=np.array(image_path_list), labels=labels, input_type=args.input_type,
                                  transform=ImageTransform(is_train=False, img_size=224))

    # Handle multiple saliency maps if needed
    if n_maps > 1:
        saliencies = []
        for n in range(n_maps):
            saliencies.append(dict())
    else:
        saliencies = dict()

    # Store values directly for creating saliency map images
    saliency_images = dict()
    original_images = dict()
    # loop over all the dataset. Can be changed to process only part of the dataset
    range_from = 0
    range_to = len(dataset)

    # loop over the dataset
    predictions = []

    labels = []
    times = []
    for i in tqdm(range(range_from, range_to), desc='generating saliency maps'):
        original_image, image, label, image_name = dataset[i]
        labels.append(int(label.item()))
        image = image.unsqueeze(0)
        if args.gpu:
            image = image.cuda()
        out = model(image)
        class_idx = out.squeeze(0).argmax().item()
        print(out.item())
        predictions.append(out.item())
        # generate saliency map depending on the choosen method
        if library == "torchcam":

            # Start measuring the time
            start_time = time.time()
            saliency_map = saliency(class_idx, out)[0]
            saliency_map = saliency_map.view(1, 1, *saliency_map.shape)
            # Measure the elapsed time
            elapsed_time = time.time() - start_time
        elif library == "rise":
            saliency_map = saliency(image)[class_idx]
        elif library == "captum":
            # Start measuring the time
            start_time = time.time()
            saliency_map = saliency.attribute(image, target=class_idx).sum(1)
            # Measure the elapsed time
            elapsed_time = time.time() - start_time
        else:
            # Start measuring the time
            start_time = time.time()
            saliency_map = saliency(image, class_idx=class_idx)
            # Measure the elapsed time
            elapsed_time = time.time() - start_time

        times.append(elapsed_time)


        if n_maps > 1:
            for n in range(n_maps):
                saliencies[n][image_name] = saliency_map[n].cpu().detach().numpy()

            saliency_images[image_name] = saliency_map[-1].cpu().squeeze(0).detach()

        else:
            saliencies[image_name] = saliency_map.cpu().detach().numpy()
            saliency_images[image_name] = saliency_map[-1].cpu().squeeze(0).detach()

        original_images[image_name] = original_image

    # Calculate the average time per iteration
    # Calculate mean and standard deviation
    mean_time = np.mean(times)
    std_time = np.std(times)

    print("Mean time:", mean_time, 's')
    print("Standard deviation:", std_time, 's')

    save_dir_npz = f"{args.save_folder}/{args.saliency}/{args.input_type}/{args.model}/npz/"
    save_dir_images = f"{args.save_folder}/{args.saliency}/{args.input_type}/{args.model}/image/"
    save_dir_saliencies = f"{args.save_folder}/{args.saliency}/{args.input_type}/{args.model}/saliency/"
    save_dir_overlays = f"{args.save_folder}/{args.saliency}/{args.input_type}/{args.model}/overlay/"
    save_dir_high_saliencies = f"{args.save_folder}/{args.saliency}/{args.input_type}/{args.model}/high_saliency/"
    Path(save_dir_npz).mkdir(parents=True, exist_ok=True)
    Path(save_dir_images).mkdir(parents=True, exist_ok=True)
    Path(save_dir_saliencies).mkdir(parents=True, exist_ok=True)
    Path(save_dir_overlays).mkdir(parents=True, exist_ok=True)
    Path(save_dir_high_saliencies).mkdir(parents=True, exist_ok=True)

    # PolyCAM methods output multiples maps for intermediate layers, export in separate files for further scripts
    print('saving images')
    if n_maps > 1:
        for n in range(n_maps):
            # Save saliency map as numpy array for later usage in the evaluation
            np.savez(save_dir_npz + args.saliency.lower() + '-' + args.input_type + '-' + args.model + '-' + args.suffix + str(n), **saliencies[n])
    else:
        np.savez(save_dir_npz + args.saliency.lower() + '-' + args.input_type + '-' + args.model + '-' + args.suffix, **saliencies)

    # Save visualization directly
    for idx, (saliency_name, saliency_map) in enumerate(saliency_images.items()):
        print(f'saving image {saliency_name}')

        if args.saliency in ["ig", "ixg", 'gb', 'sg']:
            saliency_map = torch.abs(saliency_map)

        img, saliency_img, high_activation_saliency_img, overlayed_img = overlay(original_images[saliency_name], saliency_map, alpha=0.4, colormap="turbo")
        img.save(f'{save_dir_images}/{saliency_name}_{labels[idx]}.png')
        saliency_img.save(f'{save_dir_saliencies}/{saliency_name}_{labels[idx]}.png')
        overlayed_img.save(f'{save_dir_overlays}/{saliency_name}_{labels[idx]}.png')
        high_activation_saliency_img.save(f'{save_dir_high_saliencies}/{saliency_name}_{labels[idx]}.png')
        with open(f"{args.save_folder}/{args.saliency}/{args.input_type}/{args.model}/preds.txt", 'w') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')

if __name__ == "__main__":
    main()
