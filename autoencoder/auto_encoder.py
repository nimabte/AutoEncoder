import torchvision.datasets as datasets
import torch.utils.data
import torchvision.utils
import torch.nn as nn
from torch.nn import functional as func
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
PIL_Trf=transforms.ToPILImage()

file_writer=SummaryWriter('./log')
batch_size=50
transform=transforms.ToTensor()
celeba_datasets_train=datasets.ImageFolder('./train_dataset',transform=transform)
celeba_datasets_test=datasets.ImageFolder('./test_dataset',transform=transform)
train_data_loader=torch.utils.data.DataLoader(celeba_datasets_train,batch_size=batch_size,shuffle=True)
test_data_loader=torch.utils.data.DataLoader(celeba_datasets_test,batch_size=batch_size,shuffle=True)

num_batch_train=len(celeba_datasets_train)//batch_size
num_batch_test=len(celeba_datasets_test)//batch_size
num_epochs=50
class AutoEncoderNet(nn.Module):
    def __init__(self):
        super(AutoEncoderNet, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.conv3=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1)
        self.batchnorm1=nn.BatchNorm2d(128)
        self.batchnorm2=nn.BatchNorm2d(256)
        self.batchnorm3=nn.BatchNorm2d(512)
        self.convd=nn.Conv2d(in_channels=512,out_channels=200,kernel_size=4,stride=1,padding=0)
        self.conv_transposed_tr=nn.ConvTranspose2d(in_channels=200,out_channels=512,kernel_size=4,stride=1,padding=0)
        self.conv_transpose1=nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1)
        self.conv_transpose2=nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.conv_transpose3=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.conv_transpose4=nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=4,stride=2,padding=1)
        self.batchnorm1_tr=nn.BatchNorm2d(256)
        self.batchnorm2_tr=nn.BatchNorm2d(128)
        self.batchnorm3_tr=nn.BatchNorm2d(64)

    def forward(self, x):
        x=func.leaky_relu(self.conv1(x),0.2)
        x=func.leaky_relu(self.batchnorm1(self.conv2(x)),0.2)
        x=func.leaky_relu(self.batchnorm2(self.conv3(x)),0.2)
        x=func.leaky_relu(self.batchnorm3(self.conv4(x)),0.2)
        z=self.convd(x)
        y=func.leaky_relu(self.conv_transposed_tr(z),0.2)
        y=func.leaky_relu(self.batchnorm1_tr(self.conv_transpose1(y)),.2)
        y=func.leaky_relu(self.batchnorm2_tr(self.conv_transpose2(y)),.2)
        y=func.leaky_relu(self.batchnorm3_tr(self.conv_transpose3(y)),.2)
        y=func.tanh(self.conv_transpose4(y))
        return z,y



auto_encoder_net=AutoEncoderNet()
auto_encoder_net=auto_encoder_net.cuda()
Optimizer=optim.SGD(auto_encoder_net.parameters(),lr=.001,momentum=0.95)
#lr_scheduler1=lr_scheduler.StepLR(optimizer=Optimizer,step_size=3,gamma=.1)
loss1 = nn.MSELoss(reduction='mean')

def evaluate_net(test_data_loader):
    auto_encoder_net.eval()
    test_loss=0
    for ind,data in enumerate(test_data_loader):
        test_batch,test_label=data
        test_batch=test_batch.cuda()
        test_label=test_label.cuda()
        z,y=auto_encoder_net(test_batch)
        y_out=y/2+0.5
        loss_val=loss1(y_out,test_batch)
        #print('loss val1: {0:.4f}'.format(loss_val))
        test_loss+=loss_val.cpu().item()
    test_loss=test_loss/len(test_data_loader)
    return test_loss


test_data_iter = iter(test_data_loader)
test_data,lab=next(test_data_iter)
test_data=test_data[25,:,:,:]
test_data=test_data.unsqueeze(0)
test_data=test_data.cuda()
#test_data=torchvision.utils.make_grid(test_data,padding=2)
indx_num=0
indx50=0
str_time=time.time()
print('..........training starts..........')
for ep in range(num_epochs):
    epoch_loss_train=0
    iter50_loss_train = 0
    #lr_scheduler1.step()
    for count,data in enumerate(train_data_loader):
        auto_encoder_net.train()
        Optimizer.zero_grad()
        image_batch,label=data
        image_batch=image_batch.cuda()
        label=label.cuda()
        z,y=auto_encoder_net(image_batch)
        y_out=y/2+0.5
        loss_val=loss1(y_out,image_batch)
        loss_val.backward()
        Optimizer.step()
        loss=loss_val.cpu().item()
        file_writer.add_scalar('batches_train_loss',loss,indx_num)
        indx_num+=1
        epoch_loss_train+=loss
        iter50_loss_train+=loss
        if (count%50==49) :
            end_time=time.time()
            test_loss = evaluate_net(test_data_loader)
            test_z,test_y=auto_encoder_net(test_data)
            test_y=test_y.detach().cpu()
            test_y_out=test_y/2+0.5
            test_y_out=test_y_out.squeeze(0)
            file_writer.add_image("reconstructed image", test_y_out, indx_num)
            #..................
            #file_writer.add_image('reconstructed images',test_y_out,indx50)
            print('epoch: {0:02d}/{1:02d} \t iteration: {2:03d}\{3:03d} \t train loss: {4:.4f} \t test loss: {5:.4f} \t time: {6:.2f}'\
                  .format(ep+1,num_epochs,count+1,len(train_data_loader),iter50_loss_train/50,test_loss,end_time-str_time))
            file_writer.add_scalar('train_loss',iter50_loss_train/50,indx50)
            file_writer.add_scalar('test_loss',test_loss,indx50)
            iter50_loss_train=0
            indx50+=1
            str_time=time.time()
    epoch_loss_train=epoch_loss_train/len(train_data_loader)

'''test_data_loader2=torch.utils.data.DataLoader(celeba_datasets_test,batch_size=1000,shuffle=True)
test_loader_iter2=iter(test_data_loader2)
img,labl=next(test_loader_iter2)
img=img.view(-1,200)'''
file_writer.close()
































