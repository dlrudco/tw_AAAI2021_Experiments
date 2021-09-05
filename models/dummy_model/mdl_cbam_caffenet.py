import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .cbam import CBAM


__all__ = ['CaffeNet_baseline', 'caffenet_baseline']

class CaffeNet_baseline(nn.Module):
    def __init__(self, args):
        super(CaffeNet_baseline, self).__init__()
        self.conv1_W = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2_W = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.conv3_W = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4_W = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5_W = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.fc1_W = nn.Linear(256 * 6 * 6, 4096)
        self.fc2_W = nn.Linear(4096, 4096)
        self.fc3_W = nn.Linear(4096, args.num_classes)

        self.W_params = [{'params': self.conv1_W.weight}, {'params': self.conv1_W.bias},
                         {'params': self.conv2_W.weight}, {'params': self.conv2_W.bias},
                         {'params': self.conv3_W.weight}, {'params': self.conv3_W.bias},
                         {'params': self.conv4_W.weight}, {'params': self.conv4_W.bias},
                         {'params': self.conv5_W.weight}, {'params': self.conv5_W.bias},
                         {'params': self.fc1_W.weight}, {'params': self.fc1_W.bias},
                         {'params': self.fc2_W.weight}, {'params': self.fc2_W.bias},
                         {'params': self.fc3_W.weight}, {'params': self.fc3_W.bias}]

    def forward(self, x, args):
        x = F.max_pool2d(F.relu(self.conv1_W(x * 57.6), inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x = F.local_response_norm(x, size=5, alpha=1.e-4, beta=0.75)

        x = F.max_pool2d(F.relu(self.conv2_W(x), inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x = F.local_response_norm(x, size=5, alpha=1.e-4, beta=0.75)

        x = F.relu(self.conv3_W(x), inplace=True)

        x = F.relu(self.conv4_W(x), inplace=True)

        x = F.max_pool2d(F.relu(self.conv5_W(x), inplace=True), kernel_size=3, stride=2, ceil_mode=True)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = F.relu(self.fc1_W(x), inplace=True)

        x = F.dropout(x, p=args.dropout_rate, training=self.training)

        x = F.relu(self.fc2_W(x), inplace=True)

        x = F.dropout(x, p=args.dropout_rate, training=self.training)

        x = self.fc3_W(x)

        return F.log_softmax(x, dim=1)


def caffenet_baseline(args):
    net = CaffeNet_baseline(args)
    
    model = net.state_dict()
     
    pt_model = torch.load("../pacs_final/models/alexnet_caffe.pth.tar")
        
    model['conv1_W.weight'] = pt_model['features.conv1.weight']
    model['conv1_W.bias'] = pt_model['features.conv1.bias']
    model['conv2_W.weight'] = pt_model['features.conv2.weight']
    model['conv2_W.bias'] = pt_model['features.conv2.bias']
    model['conv3_W.weight'] = pt_model['features.conv3.weight']
    model['conv3_W.bias'] = pt_model['features.conv3.bias']
    model['conv4_W.weight'] = pt_model['features.conv4.weight']
    model['conv4_W.bias'] = pt_model['features.conv4.bias']
    model['conv5_W.weight'] = pt_model['features.conv5.weight']
    model['conv5_W.bias'] = pt_model['features.conv5.bias']

    model['fc0_W.weight'] = pt_model['classifier.fc6.weight']
    model['fc0_W.bias'] = pt_model['classifier.fc6.bias']
    model['fc1_W.weight'] = pt_model['classifier.fc6.weight']
    model['fc1_W.bias'] = pt_model['classifier.fc6.bias']
    model['fc2_W.weight'] = pt_model['classifier.fc7.weight']
    model['fc2_W.bias'] = pt_model['classifier.fc7.bias']
    
    nn.init.xavier_uniform_(model['fc3_W.weight'], .1)
    nn.init.constant_(model['fc3_W.bias'], 0.)
    

    net.load_state_dict(model, strict=False)
    
    return net



# class CaffeNet_CBAM_FedDG(nn.Module):
#     def __init__(self, args):
#         super(CaffeNet_CBAM_FedDG, self).__init__()
#         self.conv1_W1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
#         self.conv1_cbam = CBAM(96, reduction_ratio=16)
#         self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
#         self.local_response_norm1 = nn.LocalResponseNorm(size=5, alpha=1.e-4, beta=0.75)

#         self.conv2_W1 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
#         self.conv2_cbam = CBAM(256, reduction_ratio=16)
#         self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
#         self.local_response_norm2 = nn.LocalResponseNorm(size=5, alpha=1.e-4, beta=0.75)

#         self.conv3_W1 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
#         self.conv3_cbam = CBAM(384, reduction_ratio=16)

#         self.conv4_W1 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
#         self.conv4_cbam = CBAM(384, reduction_ratio=16)

#         self.conv5_W1 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
#         self.conv5_cbam = CBAM(256, reduction_ratio=16)
#         self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

#         self.flatten = torch.nn.Flatten()
        
#         self.fc1_W1 = nn.Linear(256 * 6 * 6, 4096)
#         self.fc2_W1 = nn.Linear(4096, 4096)
#         self.fc3_W1 = nn.Linear(4096, args.num_classes)

#         self.log_softmax = nn.LogSoftmax(dim=1)

#         self.W1_params = [{'params': self.conv1_W1.weight}, {'params': self.conv1_W1.bias},
#                           {'params': self.conv2_W1.weight}, {'params': self.conv2_W1.bias},
#                           {'params': self.conv3_W1.weight}, {'params': self.conv3_W1.bias},
#                           {'params': self.conv4_W1.weight}, {'params': self.conv4_W1.bias},
#                           {'params': self.conv5_W1.weight}, {'params': self.conv5_W1.bias},
#                           {'params': self.fc1_W1.weight}, {'params': self.fc1_W1.bias},
#                           {'params': self.fc2_W1.weight}, {'params': self.fc2_W1.bias},
#                           {'params': self.fc3_W1.weight}, {'params': self.fc3_W1.bias}]

#         self.CBAM_params = []
#         for name, param in self.state_dict().items():
#             if 'cbam' in name:
#                 self.CBAM_params.append({'params': param})

#     def forward(self, x, args):

#         # conv1
#         x_W1 = self.conv1_W1(x * 57.6)
#         conv1_cbam = self.conv1_cbam(x_W1)
#         x = x_W1 + conv1_cbam
#         # x = conv1_cbam
#         x = self.max_pool1(F.relu(x, inplace=True))
#         x = self.local_response_norm1(x)

#         # conv2
#         x_W1 = self.conv2_W1(x)
#         conv2_cbam = self.conv2_cbam(x_W1)
#         x = x_W1 + conv2_cbam
#         # x = conv2_cbam
#         x = self.max_pool2(F.relu(x, inplace=True))
#         x = self.local_response_norm2(x)

#         # conv3
#         x_W1 = self.conv3_W1(x)
#         conv3_cbam = self.conv3_cbam(x_W1)
#         x = x_W1 + conv3_cbam
#         # x = conv3_cbam
#         x = F.relu(x, inplace=True)

#         # conv4
#         x_W1 = self.conv4_W1(x)
#         conv4_cbam = self.conv4_cbam(x_W1)
#         x = x_W1 + conv4_cbam
#         # x = conv4_cbam
#         x = F.relu(x, inplace=True)

#         # conv5
#         x_W1 = self.conv5_W1(x)
#         conv5_cbam = self.conv5_cbam(x_W1)
#         x = x_W1 + conv5_cbam
#         # x = conv5_cbam
#         x = self.max_pool5(F.relu(x, inplace=True))

#         # flatten
#         # x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.flatten(x)
#         # fc1
#         x = self.fc1_W1(x)
#         x = F.relu(x, inplace=True)
#         x = F.dropout(x, p=args.dropout_rate, training=self.training)

#         # fc2
#         x = self.fc2_W1(x)
#         x = F.relu(x, inplace=True)
#         x = F.dropout(x, p=args.dropout_rate, training=self.training)

#         # fc3
#         x = self.fc3_W1(x)
#         x = self.log_softmax(x)
#         return x

class CaffeNet_CBAM_FedDG(nn.Module):
    def __init__(self, args):
        super(CaffeNet_CBAM_FedDG, self).__init__()
        self.conv1_W1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2_W1 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.conv3_W1 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4_W1 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5_W1 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.fc1_W1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2_W1 = nn.Linear(4096, 4096)
        self.fc3_W1 = nn.Linear(4096, args.num_classes)


        self.conv1_cbam = CBAM(96, reduction_ratio=16)
        self.conv2_cbam = CBAM(256, reduction_ratio=16)
        self.conv3_cbam = CBAM(384, reduction_ratio=16)
        self.conv4_cbam = CBAM(384, reduction_ratio=16)
        self.conv5_cbam = CBAM(256, reduction_ratio=16)

        self.W1_params = [{'params': self.conv1_W1.weight}, {'params': self.conv1_W1.bias},
                          {'params': self.conv2_W1.weight}, {'params': self.conv2_W1.bias},
                          {'params': self.conv3_W1.weight}, {'params': self.conv3_W1.bias},
                          {'params': self.conv4_W1.weight}, {'params': self.conv4_W1.bias},
                          {'params': self.conv5_W1.weight}, {'params': self.conv5_W1.bias},
                          {'params': self.fc1_W1.weight}, {'params': self.fc1_W1.bias},
                          {'params': self.fc2_W1.weight}, {'params': self.fc2_W1.bias},
                          {'params': self.fc3_W1.weight}, {'params': self.fc3_W1.bias}]

        self.CBAM_params = []
        for name, param in self.state_dict().items():
            if 'cbam' in name:
                self.CBAM_params.append({'params': param})

    def forward(self, x, args):

        # conv1
        x_W1 = self.conv1_W1(x * 57.6)
        conv1_cbam = self.conv1_cbam(x_W1)
        x = x_W1 + conv1_cbam
        #x = conv1_cbam
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x = F.local_response_norm(x, size=5, alpha=1.e-4, beta=0.75)

        # conv2
        x_W1 = self.conv2_W1(x)
        conv2_cbam = self.conv2_cbam(x_W1)
        x = x_W1 + conv2_cbam
        #x = conv2_cbam
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x = F.local_response_norm(x, size=5, alpha=1.e-4, beta=0.75)

        # conv3
        x_W1 = self.conv3_W1(x)
        conv3_cbam = self.conv3_cbam(x_W1)
        x = x_W1 + conv3_cbam
        #x = conv3_cbam
        x = F.relu(x, inplace=True)

        # conv4
        x_W1 = self.conv4_W1(x)
        conv4_cbam = self.conv4_cbam(x_W1)
        x = x_W1 + conv4_cbam
        #x = conv4_cbam
        x = F.relu(x, inplace=True)

        # conv5
        x_W1 = self.conv5_W1(x)
        conv5_cbam = self.conv5_cbam(x_W1)
        x = x_W1 + conv5_cbam
        #x = conv5_cbam
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=3, stride=2, ceil_mode=True)

        # flatten
        x = x.view(x.size(0), 256 * 6 * 6)

        # fc1
        x = self.fc1_W1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=args.dropout_rate, training=self.training)

        # fc2
        x = self.fc2_W1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=args.dropout_rate, training=self.training)

        # fc3
        x = self.fc3_W1(x)
        return F.log_softmax(x, dim=1)


class CaffeNet_CBAM_Original_FedDG(nn.Module):
    def __init__(self, args):
        super(CaffeNet_CBAM_Original_FedDG, self).__init__()
        self.conv1_W1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2_W1 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.conv3_W1 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4_W1 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5_W1 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.fc1_W1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2_W1 = nn.Linear(4096, 4096)
        self.fc3_W1 = nn.Linear(4096, args.num_classes)


        self.conv1_cbam = CBAM(96, reduction_ratio=16)
        self.conv2_cbam = CBAM(256, reduction_ratio=16)
        self.conv3_cbam = CBAM(384, reduction_ratio=16)
        self.conv4_cbam = CBAM(384, reduction_ratio=16)
        self.conv5_cbam = CBAM(256, reduction_ratio=16)

        self.W1_params = [{'params': self.conv1_W1.weight}, {'params': self.conv1_W1.bias},
                          {'params': self.conv2_W1.weight}, {'params': self.conv2_W1.bias},
                          {'params': self.conv3_W1.weight}, {'params': self.conv3_W1.bias},
                          {'params': self.conv4_W1.weight}, {'params': self.conv4_W1.bias},
                          {'params': self.conv5_W1.weight}, {'params': self.conv5_W1.bias},
                          {'params': self.fc1_W1.weight}, {'params': self.fc1_W1.bias},
                          {'params': self.fc2_W1.weight}, {'params': self.fc2_W1.bias},
                          {'params': self.fc3_W1.weight}, {'params': self.fc3_W1.bias}]

        self.CBAM_params = []
        for name, param in self.state_dict().items():
            if 'cbam' in name:
                self.CBAM_params.append({'params': param})

    def forward(self, x, args):

        # conv1
        x_W1 = self.conv1_W1(x * 57.6)
        x = self.conv1_cbam(x_W1)
        # x = conv1_cbam
        x = F.max_pool2d(F.relu(x, inplace=True))
        x = F.local_response_norm(x)

        # conv2
        x_W1 = self.conv2_W1(x)
        x = self.conv2_cbam(x_W1)
        # x = conv2_cbam
        x = F.max_pool2d(F.relu(x, inplace=True))
        x = F.local_response_norm(x)

        # conv3
        x_W1 = self.conv3_W1(x)
        x = self.conv3_cbam(x_W1)
        # x = conv3_cbam
        x = F.relu(x, inplace=True)

        # conv4
        x_W1 = self.conv4_W1(x)
        x = self.conv4_cbam(x_W1)
        # x = conv4_cbam
        x = F.relu(x, inplace=True)

        # conv5
        x_W1 = self.conv5_W1(x)
        x = self.conv5_cbam(x_W1)
        # x = conv5_cbam
        x = F.max_pool2d(F.relu(x, inplace=True))

        # flatten
        x = x.view(x.size(0), 256 * 6 * 6)

        # fc1
        x = self.fc1_W1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=args.dropout_rate, training=self.training)

        # fc2
        x = self.fc2_W1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=args.dropout_rate, training=self.training)

        # fc3
        x = self.fc3_W1(x)
        return F.log_softmax(x, dim=1)

def caffenet_CBAM_FedDG(args):
    net = CaffeNet_CBAM_FedDG(args)
    
    model = net.state_dict()

    net.load_state_dict(model, strict=False)
    
    return net



def caffenet_CBAM_Original_FedDG(args):
    net = CaffeNet_CBAM_Original_FedDG(args)
    
    model = net.state_dict()

    net.load_state_dict(model, strict=False)
    
    return net
