import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


__all__ = ['CaffeNet_baseline', 'caffenet_baseline']

# class CaffeNet_baseline(nn.Module):
#     def __init__(self, args):
#         super(CaffeNet_baseline, self).__init__()
#         self.conv1_W = nn.Conv2d(3, 96, kernel_size=11, stride=4)
#         self.conv2_W = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
#         self.conv3_W = nn.Conv2d(256, 384, kernel_size=3, padding=1)
#         self.conv4_W = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
#         self.conv5_W = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
#         self.fc1_W = nn.Linear(256 * 6 * 6, 4096)
#         self.fc2_W = nn.Linear(4096, 4096)
#         self.fc3_W = nn.Linear(4096, args.num_classes)

#         self.W_params = [{'params': self.conv1_W.weight}, {'params': self.conv1_W.bias},
#                          {'params': self.conv2_W.weight}, {'params': self.conv2_W.bias},
#                          {'params': self.conv3_W.weight}, {'params': self.conv3_W.bias},
#                          {'params': self.conv4_W.weight}, {'params': self.conv4_W.bias},
#                          {'params': self.conv5_W.weight}, {'params': self.conv5_W.bias},
#                          {'params': self.fc1_W.weight}, {'params': self.fc1_W.bias},
#                          {'params': self.fc2_W.weight}, {'params': self.fc2_W.bias},
#                          {'params': self.fc3_W.weight}, {'params': self.fc3_W.bias}]

#     def forward(self, x, args):
#         x = F.max_pool2d(F.relu(self.conv1_W(x * 57.6), inplace=True), kernel_size=3, stride=2, ceil_mode=True)
#         x = F.local_response_norm(x, size=5, alpha=1.e-4, beta=0.75)

#         x = F.max_pool2d(F.relu(self.conv2_W(x), inplace=True), kernel_size=3, stride=2, ceil_mode=True)
#         x = F.local_response_norm(x, size=5, alpha=1.e-4, beta=0.75)

#         x = F.relu(self.conv3_W(x), inplace=True)

#         x = F.relu(self.conv4_W(x), inplace=True)

#         x = F.max_pool2d(F.relu(self.conv5_W(x), inplace=True), kernel_size=3, stride=2, ceil_mode=True)

#         x = x.view(x.size(0), 256 * 6 * 6)

#         x = F.relu(self.fc1_W(x), inplace=True)

#         x = F.dropout(x, p=args.dropout_rate, training=self.training)

#         x = F.relu(self.fc2_W(x), inplace=True)

#         x = F.dropout(x, p=args.dropout_rate, training=self.training)

#         x = self.fc3_W(x)

#         return F.log_softmax(x, dim=1)

class CaffeNet_baseline(nn.Module):
    def __init__(self, args):
        super(CaffeNet_baseline, self).__init__()

        self.conv1_W = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.local_response_norm1 = nn.LocalResponseNorm(size=5, alpha=1.e-4, beta=0.75)

        self.conv2_W = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.local_response_norm2 = nn.LocalResponseNorm(size=5, alpha=1.e-4, beta=0.75)

        self.conv3_W = nn.Conv2d(256, 384, kernel_size=3, padding=1)

        self.conv4_W = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)

        self.conv5_W = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.flatten = torch.nn.Flatten()
        
        self.fc1_W = nn.Linear(256 * 6 * 6, 4096)

        self.fc2_W = nn.Linear(4096, 4096)

        self.fc3_W = nn.Linear(4096, args.num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.W_params = [{'params': self.conv1_W.weight}, {'params': self.conv1_W.bias},
                         {'params': self.conv2_W.weight}, {'params': self.conv2_W.bias},
                         {'params': self.conv3_W.weight}, {'params': self.conv3_W.bias},
                         {'params': self.conv4_W.weight}, {'params': self.conv4_W.bias},
                         {'params': self.conv5_W.weight}, {'params': self.conv5_W.bias},
                         {'params': self.fc1_W.weight}, {'params': self.fc1_W.bias},
                         {'params': self.fc2_W.weight}, {'params': self.fc2_W.bias},
                         {'params': self.fc3_W.weight}, {'params': self.fc3_W.bias}]

    def forward(self, x, args, feature_extract=False):
        x = F.relu(self.conv1_W(x * 57.6), inplace=True)
        x = self.max_pool1(x)
        x = self.local_response_norm1(x)

        x = F.relu(self.conv2_W(x), inplace=True)
        x = self.max_pool2(x)
        x = self.local_response_norm2(x)

        x = F.relu(self.conv3_W(x), inplace=True)

        x = self.conv4_W(x)
        x = F.relu(x, inplace=True)

        x = F.relu(self.conv5_W(x), inplace=True)
        x = self.max_pool5(x)

        x = self.flatten(x)
        x = F.relu(self.fc1_W(x), inplace=True)

        # x = F.dropout(x, p=args.dropout_rate, training=self.training)
        x = F.relu(self.fc2_W(x), inplace=True)

        # x = F.dropout(x, p=args.dropout_rate, training=self.training)

        x = self.fc3_W(x)
        x = self.log_softmax(x)

        return x

def caffenet_baseline(args):
    net = CaffeNet_baseline(args)
    
    model = net.state_dict()
    '''     
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

    model['fc1_W.weight'] = pt_model['classifier.fc6.weight']
    model['fc1_W.bias'] = pt_model['classifier.fc6.bias']
    model['fc2_W.weight'] = pt_model['classifier.fc7.weight']
    model['fc2_W.bias'] = pt_model['classifier.fc7.bias']
    
    nn.init.xavier_uniform_(model['fc3_W.weight'], .1)
    nn.init.constant_(model['fc3_W.bias'], 0.)
    '''

    net.load_state_dict(model, strict=False)
    
    return net





class CaffeNet_FedDG(nn.Module):
    def __init__(self, args):
        super(CaffeNet_FedDG, self).__init__()
        self.conv1_W1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2_W1 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.conv3_W1 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4_W1 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5_W1 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.fc1_W1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2_W1 = nn.Linear(4096, 4096)
        self.fc3_W1 = nn.Linear(4096, args.num_classes)

        self.conv1_W2 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2_W2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.conv3_W2 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4_W2 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5_W2 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.fc1_W2 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2_W2 = nn.Linear(4096, 4096)
        self.fc3_W2 = nn.Linear(4096, args.num_classes)

        self.W1_params = [{'params': self.conv1_W1.weight}, {'params': self.conv1_W1.bias},
                          {'params': self.conv2_W1.weight}, {'params': self.conv2_W1.bias},
                          {'params': self.conv3_W1.weight}, {'params': self.conv3_W1.bias},
                          {'params': self.conv4_W1.weight}, {'params': self.conv4_W1.bias},
                          {'params': self.conv5_W1.weight}, {'params': self.conv5_W1.bias},
                          {'params': self.fc1_W1.weight}, {'params': self.fc1_W1.bias},
                          {'params': self.fc2_W1.weight}, {'params': self.fc2_W1.bias},
                          {'params': self.fc3_W1.weight}, {'params': self.fc3_W1.bias}]

        self.W2_params = [{'params': self.conv1_W2.weight}, {'params': self.conv1_W2.bias},
                          {'params': self.conv2_W2.weight}, {'params': self.conv2_W2.bias},
                          {'params': self.conv3_W2.weight}, {'params': self.conv3_W2.bias},
                          {'params': self.conv4_W2.weight}, {'params': self.conv4_W2.bias},
                          {'params': self.conv5_W2.weight}, {'params': self.conv5_W2.bias},
                          {'params': self.fc1_W2.weight}, {'params': self.fc1_W2.bias},
                          {'params': self.fc2_W2.weight}, {'params': self.fc2_W2.bias},
                          {'params': self.fc3_W2.weight}, {'params': self.fc3_W2.bias}]
    '''
    def forward(self, x, args):
        x_W1 = self.conv1_W1(x * 57.6)
        x_W2 = self.conv1_W2(x * 57.6)
        #x = x_W1 + x_W2
        x_W1 = F.max_pool2d(F.relu(x_W1, inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x_W1 = F.local_response_norm(x_W1, size=5, alpha=1.e-4, beta=0.75)
        x_W2 = F.max_pool2d(F.relu(x_W2, inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x_W2 = F.local_response_norm(x_W2, size=5, alpha=1.e-4, beta=0.75)

        x_W1 = self.conv2_W1(x_W1)
        x_W2 = self.conv2_W2(x_W2)
        #x = x_W1 + x_W2
        x_W1 = F.max_pool2d(F.relu(x_W1, inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x_W1 = F.local_response_norm(x_W1, size=5, alpha=1.e-4, beta=0.75)
        x_W2 = F.max_pool2d(F.relu(x_W2, inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x_W2 = F.local_response_norm(x_W2, size=5, alpha=1.e-4, beta=0.75)

        x_W1 = self.conv3_W1(x_W1)
        x_W2 = self.conv3_W2(x_W2)
        #x = x_W1 + x_W2
        x_W1 = F.relu(x_W1, inplace=True)
        x_W2 = F.relu(x_W2, inplace=True)

        x_W1 = self.conv4_W1(x_W1)
        x_W2 = self.conv4_W2(x_W2)
        #x = x_W1 + x_W2
        x_W1 = F.relu(x_W1, inplace=True)
        x_W2 = F.relu(x_W2, inplace=True)

        x_W1 = self.conv5_W1(x_W1)
        x_W2 = self.conv5_W2(x_W2)
        #x = x_W1 + x_W2
        x_W1 = F.max_pool2d(F.relu(x_W1, inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x_W2 = F.max_pool2d(F.relu(x_W2, inplace=True), kernel_size=3, stride=2, ceil_mode=True)

        x_W1 = x_W1.view(x_W1.size(0), 256 * 6 * 6)
        x_W2 = x_W2.view(x_W2.size(0), 256 * 6 * 6)

        x_W1 = self.fc1_W1(x_W1)
        x_W2 = self.fc1_W2(x_W2)
        #x = x_W1 + x_W2
        x_W1 = F.relu(x_W1, inplace=True)
        x_W1 = F.dropout(x_W1, p=args.dropout_rate, training=self.training)
        x_W2 = F.relu(x_W2, inplace=True)
        x_W2 = F.dropout(x_W2, p=args.dropout_rate, training=self.training)

        x_W1 = self.fc2_W1(x_W1)
        x_W2 = self.fc2_W2(x_W2)
        #x = x_W1 + x_W2
        x_W1 = F.relu(x_W1, inplace=True)
        x_W1 = F.dropout(x_W1, p=args.dropout_rate, training=self.training)
        x_W2 = F.relu(x_W2, inplace=True)
        x_W2 = F.dropout(x_W2, p=args.dropout_rate, training=self.training)

        x_W1 = self.fc3_W1(x_W1)
        x_W2 = self.fc3_W2(x_W2)
        x = x_W1 + x_W2

        return F.log_softmax(x, dim=1)
    '''
    
    def forward(self, x, args):
        x = self.conv1_W1(x * 57.6)
        #x_W1 = self.conv1_W1(x * 57.6)
        #x_W2 = self.conv1_W2(x * 57.6)
        #x = (x_W1 + x_W2) * 1.0
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x = F.local_response_norm(x, size=5, alpha=1.e-4, beta=0.75)

        x = self.conv2_W1(x)
        #x_W1 = self.conv2_W1(x)
        #x_W2 = self.conv2_W2(x)
        #x = (x_W1 + x_W2) * 1.0
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=3, stride=2, ceil_mode=True)
        x = F.local_response_norm(x, size=5, alpha=1.e-4, beta=0.75)

        x = self.conv3_W1(x)
        #x_W1 = self.conv3_W1(x)
        #x_W2 = self.conv3_W2(x)
        #x = (x_W1 + x_W2) * 1.0
        x = F.relu(x, inplace=True)

        x = self.conv4_W1(x)
        #x_W1 = self.conv4_W1(x)
        #x_W2 = self.conv4_W2(x)
        #x = (x_W1 + x_W2) * 1.0
        x = F.relu(x, inplace=True)

        x = self.conv5_W1(x)
        #x_W1 = self.conv5_W1(x)
        #x_W2 = self.conv5_W2(x)
        #x = (x_W1 + x_W2) * 1.0
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=3, stride=2, ceil_mode=True)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.fc1_W1(x)
        #x_W1 = self.fc1_W1(x)
        #x_W2 = self.fc1_W2(x)
        #x = (x_W1 + x_W2) * 1.0
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=args.dropout_rate, training=self.training)

        x = self.fc2_W1(x)
        #x_W1 = self.fc2_W1(x)
        #x_W2 = self.fc2_W2(x)
        #x = (x_W1 + x_W2) * 1.0
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=args.dropout_rate, training=self.training)

        x = self.fc3_W1(x)
        #x_W1 = self.fc3_W1(x)
        #x_W2 = self.fc3_W2(x)
        #x = (x_W1 + x_W2) * 1.0

        return F.log_softmax(x, dim=1)
    

def caffenet_FedDG(args):
    net = CaffeNet_FedDG(args)
    
    model = net.state_dict()
    '''
    model['conv1_W1.weight'] = model['conv1_W1.weight'] * 0.5
    model['conv1_W1.bias'] = model['conv1_W1.bias'] * 0.5
    model['conv2_W1.weight'] = model['conv2_W1.weight'] * 0.5
    model['conv2_W1.bias'] = model['conv2_W1.bias'] * 0.5
    model['conv3_W1.weight'] = model['conv3_W1.weight'] * 0.5
    model['conv3_W1.bias'] = model['conv3_W1.bias'] * 0.5
    model['conv4_W1.weight'] = model['conv4_W1.weight'] * 0.5
    model['conv4_W1.bias'] = model['conv4_W1.bias'] * 0.5
    model['conv5_W1.weight'] = model['conv5_W1.weight'] * 0.5
    model['conv5_W1.bias'] = model['conv5_W1.bias'] * 0.5

    model['fc1_W1.weight'] = model['fc1_W1.weight'] * 0.5
    model['fc1_W1.bias'] = model['fc1_W1.bias'] * 0.5
    model['fc2_W1.weight'] = model['fc2_W1.weight'] * 0.5
    model['fc2_W1.bias'] = model['fc2_W1.bias'] * 0.5
    model['fc3_W1.weight'] = model['fc3_W1.weight'] * 0.5
    model['fc3_W1.bias'] = model['fc3_W1.bias'] * 0.5

    model['conv1_W2.weight'] = model['conv1_W2.weight'] * 0.5
    model['conv1_W2.bias'] = model['conv1_W2.bias'] * 0.5
    model['conv2_W2.weight'] = model['conv2_W2.weight'] * 0.5
    model['conv2_W2.bias'] = model['conv2_W2.bias'] * 0.5
    model['conv3_W2.weight'] = model['conv3_W2.weight'] * 0.5
    model['conv3_W2.bias'] = model['conv3_W2.bias'] * 0.5
    model['conv4_W2.weight'] = model['conv4_W2.weight'] * 0.5
    model['conv4_W2.bias'] = model['conv4_W2.bias'] * 0.5
    model['conv5_W2.weight'] = model['conv5_W2.weight'] * 0.5
    model['conv5_W2.bias'] = model['conv5_W2.bias'] * 0.5

    model['fc1_W2.weight'] = model['fc1_W2.weight'] * 0.5
    model['fc1_W2.bias'] = model['fc1_W2.bias'] * 0.5
    model['fc2_W2.weight'] = model['fc2_W2.weight'] * 0.5
    model['fc2_W2.bias'] = model['fc2_W2.bias'] * 0.5
    model['fc3_W2.weight'] = model['fc3_W2.weight'] * 0.5
    model['fc3_W2.bias'] = model['fc3_W2.bias'] * 0.5
    '''
        
    pt_model = torch.load("../pacs_final/models/alexnet_caffe.pth.tar")
    '''
    model['conv1_W1.weight'] = pt_model['features.conv1.weight'] * 0.5
    model['conv1_W1.bias'] = pt_model['features.conv1.bias'] * 0.5
    model['conv2_W1.weight'] = pt_model['features.conv2.weight'] * 0.5
    model['conv2_W1.bias'] = pt_model['features.conv2.bias'] * 0.5
    model['conv3_W1.weight'] = pt_model['features.conv3.weight'] * 0.5
    model['conv3_W1.bias'] = pt_model['features.conv3.bias'] * 0.5
    model['conv4_W1.weight'] = pt_model['features.conv4.weight'] * 0.5
    model['conv4_W1.bias'] = pt_model['features.conv4.bias'] * 0.5
    model['conv5_W1.weight'] = pt_model['features.conv5.weight'] * 0.5
    model['conv5_W1.bias'] = pt_model['features.conv5.bias'] * 0.5

    model['fc1_W1.weight'] = pt_model['classifier.fc6.weight'] * 0.5
    model['fc1_W1.bias'] = pt_model['classifier.fc6.bias'] * 0.5
    model['fc2_W1.weight'] = pt_model['classifier.fc7.weight'] * 0.5
    model['fc2_W1.bias'] = pt_model['classifier.fc7.bias'] * 0.5


    model['conv1_W2.weight'] = pt_model['features.conv1.weight'] * 0.5
    model['conv1_W2.bias'] = pt_model['features.conv1.bias'] * 0.5
    model['conv2_W2.weight'] = pt_model['features.conv2.weight'] * 0.5
    model['conv2_W2.bias'] = pt_model['features.conv2.bias'] * 0.5
    model['conv3_W2.weight'] = pt_model['features.conv3.weight'] * 0.5
    model['conv3_W2.bias'] = pt_model['features.conv3.bias'] * 0.5
    model['conv4_W2.weight'] = pt_model['features.conv4.weight'] * 0.5
    model['conv4_W2.bias'] = pt_model['features.conv4.bias'] * 0.5
    model['conv5_W2.weight'] = pt_model['features.conv5.weight'] * 0.5
    model['conv5_W2.bias'] = pt_model['features.conv5.bias'] * 0.5

    model['fc1_W2.weight'] = pt_model['classifier.fc6.weight'] * 0.5
    model['fc1_W2.bias'] = pt_model['classifier.fc6.bias'] * 0.5
    model['fc2_W2.weight'] = pt_model['classifier.fc7.weight'] * 0.5
    model['fc2_W2.bias'] = pt_model['classifier.fc7.bias'] * 0.5
    '''

        
    model['conv1_W1.weight'] = pt_model['features.conv1.weight']
    model['conv1_W1.bias'] = pt_model['features.conv1.bias']
    model['conv2_W1.weight'] = pt_model['features.conv2.weight']
    model['conv2_W1.bias'] = pt_model['features.conv2.bias']
    model['conv3_W1.weight'] = pt_model['features.conv3.weight']
    model['conv3_W1.bias'] = pt_model['features.conv3.bias']
    model['conv4_W1.weight'] = pt_model['features.conv4.weight']
    model['conv4_W1.bias'] = pt_model['features.conv4.bias']
    model['conv5_W1.weight'] = pt_model['features.conv5.weight']
    model['conv5_W1.bias'] = pt_model['features.conv5.bias']

    model['fc1_W1.weight'] = pt_model['classifier.fc6.weight']
    model['fc1_W1.bias'] = pt_model['classifier.fc6.bias']
    model['fc2_W1.weight'] = pt_model['classifier.fc7.weight']
    model['fc2_W1.bias'] = pt_model['classifier.fc7.bias']

    
    model['conv1_W2.weight'] = pt_model['features.conv1.weight']
    model['conv1_W2.bias'] = pt_model['features.conv1.bias']
    model['conv2_W2.weight'] = pt_model['features.conv2.weight']
    model['conv2_W2.bias'] = pt_model['features.conv2.bias']
    model['conv3_W2.weight'] = pt_model['features.conv3.weight']
    model['conv3_W2.bias'] = pt_model['features.conv3.bias']
    model['conv4_W2.weight'] = pt_model['features.conv4.weight']
    model['conv4_W2.bias'] = pt_model['features.conv4.bias']
    model['conv5_W2.weight'] = pt_model['features.conv5.weight']
    model['conv5_W2.bias'] = pt_model['features.conv5.bias']

    model['fc1_W2.weight'] = pt_model['classifier.fc6.weight'] * 0.5
    model['fc1_W2.bias'] = pt_model['classifier.fc6.bias'] * 0.5
    model['fc2_W2.weight'] = pt_model['classifier.fc7.weight']
    model['fc2_W2.bias'] = pt_model['classifier.fc7.bias']
     
    
    #1
    nn.init.xavier_uniform_(model['fc3_W1.weight'], .1)
    nn.init.constant_(model['fc3_W1.bias'], 0.)
    nn.init.xavier_uniform_(model['fc3_W2.weight'], .1)
    nn.init.constant_(model['fc3_W2.bias'], 0.)

    #2
    #nn.init.xavier_uniform_(model['fc3_W1.weight'], .1)
    #nn.init.constant_(model['fc3_W1.bias'], 0.) 
    #model['fc3_W2.weight'] = model['fc3_W1.weight'] * 0.5
    #model['fc3_W2.bias'] = model['fc3_W1.bias'] * 0.5
    #model['fc3_W1.weight'] = model['fc3_W1.weight'] * 0.5
    #model['fc3_W1.bias'] = model['fc3_W1.bias'] * 0.5

    net.load_state_dict(model, strict=False)
    
    return net







