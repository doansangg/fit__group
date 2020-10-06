from test import *
from process.augmentation import *
from process.data import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='baseline')
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--select', type=str, default="color")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cycle_num', type=int, default=10)
parser.add_argument('--cycle_inter', type=int, default=50)
parser.add_argument('--image_folder', type=str, default=None)
parser.add_argument('--pretrained_model', type=str, default=None)
parser.add_argument('--save_file', type=str, default=None)
config = parser.parse_args()
def get_augment(image_mode):
    if image_mode == 'color':
        augment = color_augumentor
    elif image_mode == 'depth':
        augment = depth_augumentor
    elif image_mode == 'ir':
        augment = ir_augumentor
    return augment
augment = get_augment(config.select)
valid_dataset= FDDataset(config.image_folder,config.image_size,augment=augment,save_file_txt=config.save_file)
valid_loader  = DataLoader( valid_dataset,shuffle=False,batch_size=2,drop_last   = False,num_workers=8)
criterion = softmax_cross_entropy_criterion
initial_checkpoint=config.pretrained_model+"/global_min_acer_model.pth"
net = torch.nn.DataParallel(net)
net =  net.cuda()
n=torch.load(initial_checkpoint)
net.load_state_dict(n)
net.eval()

#for k in valid_dataset:
#    print(k)
#print(valid_dataset)
out = infer_test(net, valid_loader)
print(out)
file2=open(config.save_file+'/test1.txt','w+')
file1=open(config.save_file+'/test.txt','r')
#valid_loss,out = do_valid_test(net, valid_loader, criterion)
k=file1.readlines()
for i in range(len(out)):
	m=k[i].split("\n")[0]+"      "+str(out[i])+'\n'
	file2.write(m)
file1.close()
file2.close()
