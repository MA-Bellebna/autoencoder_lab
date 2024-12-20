import torch 
import torch.nn as nn
import torch.nn.functional as F
from network.model_mobilefacenet import MobileFaceNet
import torchvision.transforms as transforms
center_crop = transforms.CenterCrop(112)


class Facelatent(nn.Module):
    def __init__(self,filters):
        super(Facelatent, self).__init__()

        f = filters 
        # f = [960, 480, 240, 120, 60]
        self.conv1 = nn.Conv2d(in_channels=220, 
                            out_channels=f[0],
                            kernel_size=3,
                            stride=1, padding=1
                            ,padding_mode='reflect'
                                )
        self.bn1 = nn.BatchNorm2d(f[0])

        self.conv2 = nn.Conv2d(in_channels=f[0]
                                ,out_channels=f[1], 
                                kernel_size=3, 
                                stride=1, 
                                padding=1, 
                                padding_mode='reflect'
                                )
        
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.bn2 = nn.BatchNorm2d(f[1])

        self.conv3 = nn.Conv2d(in_channels=f[1],
                                out_channels=f[2],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode='reflect')
        self.bn3 = nn.BatchNorm2d(f[2])

        self.conv4 = nn.Conv2d(in_channels=f[2],
                                out_channels=f[3],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode='reflect')
        self.bn4 = nn.BatchNorm2d(f[3])

        self.conv5 = nn.Conv2d(in_channels=f[3],
                                out_channels=f[4],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode='reflect')
        
        self.bn5 = nn.BatchNorm2d(f[4])


        self.convx = nn.Conv2d(in_channels=f[4],
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode='reflect')



    def forward(self, input):

        output = F.relu(self.bn1(self.conv1(input)))  
        output = self.upscale(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.upscale(output)
        output = F.relu(self.bn3(self.conv3(output)))
        output = self.upscale(output)
        output = F.relu(self.bn4(self.conv4(output))) 
        output = self.upscale(output)
        output = F.relu(self.bn5(self.conv5(output)))  
        output = self.upscale(output)
        output = self.convx(output)
        
        return output
    



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        # out = out * 0.2
        out += x
        
        return out


class FaceNetBlock(nn.Module):
    def __init__(self, num_blocks, channels):
        super(FaceNetBlock, self).__init__()

        self.blocks = nn.ModuleList([ResidualBlock(channels, channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(channels, channels*4, kernel_size = 3, stride = 1, padding = 1)
        # self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscale = nn.PixelShuffle(2)
        # self.upscale = nn.ConvTranspose2d(channels, channels, 2, stride=2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.conv(x)
        x = self.upscale(x)
        return x
    

class Body(nn.Module):
    def __init__(self, num_blocks, facenet_blocks, channels):
        super(Body, self).__init__()
        
        self.blocks = nn.ModuleList([FaceNetBlock(facenet_blocks, channels) for _ in range(num_blocks)])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FaceDecoder(nn.Module):
    def __init__(self, num_blocks, channels):
        super(FaceDecoder, self).__init__()

        self.conv0 = nn.Conv2d(220, channels, kernel_size = 3, padding = 1, stride = 1)
        self.bn0 = nn.BatchNorm2d(channels)
        self.relu0 = nn.ReLU()
        self.body  = Body(num_blocks, facenet_blocks=2, channels = channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, stride = 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, 3, kernel_size = 3, padding = 1, stride = 1)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        y = self.body(x)
        x = self.conv1(y)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    
if __name__ == "__main__":

    # model = Facelatent(filters=[400, 360, 240, 120, 60])
    model = FaceDecoder(4, 128).to(torch.device("cpu"))
    y = torch.rand((1,220,8,8)).to(torch.device("cpu"))

    def load_mobileface():
        test = MobileFaceNet(512)

        load = torch.load(r'../models/MobileFace_Net')
        test.load_state_dict(load)

        for param in model.parameters():
            param.requires_grad = False

        return test

    # total_params = sum(
	# param.numel() for param in model.parameters())
    # print(total_params)
    z= model(y)
    print(z.shape)
    z = center_crop(z)
    test = load_mobileface().to(torch.device("cpu"))
    test.eval()
    print(z.shape)
    emb = test(z)
    print(emb.shape)
    # print(test)
    # print(model)
    # >> 131068