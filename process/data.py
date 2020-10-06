from utils import *
import cv2
import os
from process.augmentation import *
from process.data_helper import *
class FDDataset(Dataset):
    def __init__(self,image_folder, image_size=128,augment = None,save_file_txt=None):
        super(FDDataset, self).__init__()

        self.channels = 3
        self.image_size = image_size
        self.image_folder=image_folder
        self.augment=augment
        self.save_file_txt=save_file_txt
        self.test_list=[]
        self.image_list=[]
        self.set_mode(self.image_folder,self.save_file_txt)
    def set_mode(self, image_folder,save_file_txt):
        self.image_folder=image_folder
        self.save_file_txt=save_file_txt
        self.image_list=os.listdir(self.image_folder)
        self.test_list=[]
        file1 = open(self.save_file_txt+'/test.txt',"w+") 
        for x in self.image_list:
            self.test_list.append(os.path.join(self.image_folder,x))
            string=x+'\n'
            file1.write(string)
        file1.close()
        self.num_data = len(self.test_list)
        print('set dataset mode: test')

        print(self.num_data)

    def __getitem__(self, index):


        ir = self.test_list[index]
        print(ir)
        test_id = ' ' + ir
        image = cv2.imread(ir,1)
        image = cv2.resize(image,(112,112))

        image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer = True)
        n = len(image)
        image = np.concatenate(image,axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        image = image.astype(np.float32)
        image = image.reshape([n, self.channels, self.image_size, self.image_size])
        image = image / 255.0

        return (torch.FloatTensor(image), test_id)


    def __len__(self):
        return self.num_data



