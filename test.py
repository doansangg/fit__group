import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0' #'3,2,1,0'
import sys
sys.path.append("..")
import argparse
from process.data import *
from process.augmentation import *
from metric import *
model_name = "model_A"
def get_model(model_name, num_class,is_first_bn):
    if model_name == 'baseline':
        from model.model_baseline import Net
    elif model_name == 'model_A':
        from model.FaceBagNet_model_A import Net
    elif model_name == 'model_B':
        from model.FaceBagNet_model_B import Net
    elif model_name == 'model_C':
        from model.FaceBagNet_model_C import Net

    net = Net(num_class=num_class,is_first_bn=is_first_bn)
    return net
net = get_model(model_name, num_class=2, is_first_bn=True)
print(net)
