import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from models import CaffeNet_baseline

import os
import pickle

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if name in self.target_layers:
                x = module(x)
                # print("Hello")
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.target_layers = target_layers
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            # print(name, self.target_layers, name in self.target_layers, module == self.feature_module)
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        # breakpoint()
        return target_activations, x


def preprocess_image(img):
    means = [0.3675, 0.3803, 0.3394]
    stds = [0.1489, 0.1392, 0.1346]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    inp = preprocessed_img.requires_grad_(True)
    return inp


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, inp):
        return self.model(inp, args)

    def __call__(self, inp, index=None):
        if self.cuda:
            features, output = self.extractor(inp.cuda())
        else:
            features, output = self.extractor(inp)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        # breakpoint()
        # self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, inp.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, inp):
        positive_mask = (inp > 0).type_as(inp)
        output = torch.addcmul(torch.zeros(inp.size()).type_as(inp), inp, positive_mask)
        self.save_for_backward(inp, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        inp, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (inp > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(inp.size()).type_as(inp),
                                   torch.addcmul(torch.zeros(inp.size()).type_as(inp), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, inp):
        return self.model(inp, args)

    def __call__(self, inp, index=None):
        if self.cuda:
            output = self.forward(inp.cuda())
        else:
            output = self.forward(inp)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = inp.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--folder_path', type=str, default='./images/dummy_dataset/val/',
                        help='Input image path')
    parser.add_argument('--model_name', type=str, default='dummy_model',
                        help='target model name')
    parser.add_argument('--out_path', type=str, default='./gradcams/',
                        help='Input image path')
    parser.add_argument('--resume', type=str, default='checkpoints/dummy_model/dummy_model.pth.tar',
                        help='Input image path')
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    global args
    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50(pretrained=True)
    # model = caffenet_CBAM_FedDG(args)
    model = CaffeNet_baseline(args)
    # print(model)
    print(f"=> loading checkpoint {args.resume}")
    checkpoint = torch.load(args.resume)

    model.load_state_dict(checkpoint['state_dict'])

    print(f"=> loaded checkpoint {args.resume}")
    # breakpoint()

    model.eval()

    grad_cam = GradCam(model=model, feature_module=model.conv5_W, \
                       target_layer_names=["conv5_W"], use_cuda=args.use_cuda)

    sub_folder_list = os.listdir(args.folder_path)
    os.makedirs(args.out_path, exist_ok=True)
    
    for sub_folder in sub_folder_list:
        print(sub_folder)
        # os.makedirs(f'{args.out_path}/{sub_folder}/gb', exist_ok=True)
        os.makedirs(f'{args.out_path}/{args.model_name}/{sub_folder}/cam', exist_ok=True)
        os.makedirs(f'{args.out_path}/{args.model_name}/{sub_folder}/vis', exist_ok=True)
        image_list = os.listdir(f'{args.folder_path}/{sub_folder}')
        for image_path in image_list:
            img = cv2.imread(f'{args.folder_path}/{sub_folder}/'+image_path, 1)
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            inp = preprocess_image(img)
            image_name = image_path.split('.')[0]
            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested index.
            target_index = None
            mask = grad_cam(inp, target_index)
            pickle.dump(mask, open(f'{args.out_path}/{args.model_name}/{sub_folder}/cam/{image_name}.pkl','wb'))
            # show_cam_on_image(img, mask)
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            image_name = image_path.split('.')[0]
            # cv2.imwrite(f'{args.out_path}/{sub_folder}/gb/{image_name}.jpg', gb)
            cv2.imwrite(f'{args.out_path}/{args.model_name}/{sub_folder}/vis/{image_name}.jpg', np.uint8(255 * cam))


            