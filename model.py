import torch
import torch.nn as nn
import torch.nn.functional as F

class ERSTEN(nn.Module):
    def __init__(self, nc, input_size, extract_chn, param1=None, param2=None, param3=None, param4=None, num_classes=43):
        super(ERSTEN, self).__init__()

        self.extract_chn = extract_chn
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4
        self.input_size = input_size
        self.nc = nc

        # Extractor
        self.ex_pd1 = nn.ReplicationPad2d(2)
        self.ex1 = nn.Conv2d(nc, self.extract_chn[0], 5, 1)
        self.ex_bn1 = nn.InstanceNorm2d(self.extract_chn[0])

        self.ex_pd2 = nn.ReplicationPad2d(2)
        self.ex2 = nn.Conv2d(self.extract_chn[0], self.extract_chn[1], 5, 1)
        self.ex_bn2 = nn.InstanceNorm2d(self.extract_chn[1])

        self.ex_pd3 = nn.ReplicationPad2d(1)
        self.ex3 = nn.Conv2d(self.extract_chn[1], self.extract_chn[2], 3, 1)
        self.ex_bn3 = nn.InstanceNorm2d(self.extract_chn[2])

        self.ex_pd4 = nn.ReplicationPad2d(1)
        self.ex4 = nn.Conv2d(self.extract_chn[2], self.extract_chn[3], 3, 1)
        self.ex_bn4 = nn.InstanceNorm2d(self.extract_chn[3])

        self.ex_pd5 = nn.ReplicationPad2d(1)
        self.ex5 = nn.Conv2d(self.extract_chn[3], self.extract_chn[4], 3, 1)
        self.ex_bn5 = nn.InstanceNorm2d(self.extract_chn[4])

        self.ex_pd6 = nn.ReplicationPad2d(1)
        self.ex6 = nn.Conv2d(self.extract_chn[4], self.extract_chn[5], 3, 1)
        self.ex_bn6 = nn.InstanceNorm2d(self.extract_chn[5])

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Decoder
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.de_pd1 = nn.ReplicationPad2d(1)
        self.de1 = nn.Conv2d(int(self.extract_chn[5] / 2), self.extract_chn[4], 3, 1)
        self.de_bn1 = nn.InstanceNorm2d(self.extract_chn[4], 1.e-3)

        self.de_pd2 = nn.ReplicationPad2d(1)
        self.de2 = nn.Conv2d(self.extract_chn[4], self.extract_chn[3], 3, 1)
        self.de_bn2 = nn.InstanceNorm2d(self.extract_chn[3], 1.e-3)

        self.de_pd3 = nn.ReplicationPad2d(1)
        self.de3 = nn.Conv2d(self.extract_chn[3], self.extract_chn[2], 3, 1)
        self.de_bn3 = nn.InstanceNorm2d(self.extract_chn[2], 1.e-3)

        self.de_pd4 = nn.ReplicationPad2d(1)
        self.de4 = nn.Conv2d(self.extract_chn[2], self.extract_chn[1], 3, 1)
        self.de_bn4 = nn.InstanceNorm2d(self.extract_chn[1], 1.e-3)

        self.de_pd5 = nn.ReplicationPad2d(1)
        self.de5 = nn.Conv2d(self.extract_chn[1], nc, 3, 1)

        # Warping
        if param1 is not None:
            self.stn1 = stn(nc, self.input_size, param1)
        if param2 is not None:
            self.stn2 = stn(self.extract_chn[1], self.input_size, param2)
        if param3 is not None:
            self.stn3 = stn(self.extract_chn[3], self.input_size, param3)
        if param4 is not None:
            self.stn4 = stn(int(self.extract_chn[5] / 2), self.input_size, param4)

        # Classifier (from Code 2)
        self.conv1 = nn.Conv2d(in_channels=self.extract_chn[5], out_channels=128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.25)

    def extract(self, x, is_warping):
        if is_warping and self.param1 is not None:
            x = self.stn1(x)
        h1 = self.leakyrelu(self.ex_bn1(self.ex1(self.ex_pd1(x))))
        h2 = self.leakyrelu(self.ex_bn2(self.ex2(self.ex_pd2(h1))))

        if is_warping and self.param2 is not None:
            h2 = self.stn2(h2)
        h3 = self.leakyrelu(self.ex_bn3(self.ex3(self.ex_pd3(h2))))
        h4 = self.leakyrelu(self.ex_bn4(self.ex4(self.ex_pd4(h3))))

        if is_warping and self.param3 is not None:
            h4 = self.stn3(h4)
        h5 = self.leakyrelu(self.ex_bn5(self.ex5(self.ex_pd5(h4))))
        h6 = self.sigmoid(self.ex_bn6(self.ex6(self.ex_pd6(h5))))

        feat_sem, feat_illu = torch.chunk(h6, 2, 1)
        return feat_sem, feat_illu

    def decode(self, x):
        h1 = self.leakyrelu(self.de_bn1(self.de1(self.de_pd1(x))))
        h2 = self.leakyrelu(self.de_bn2(self.de2(self.de_pd2(h1))))
        h3 = self.leakyrelu(self.de_bn3(self.de3(self.de_pd3(h2))))
        h4 = self.leakyrelu(self.de_bn4(self.de4(self.de_pd4(h3))))
        out = self.sigmoid(self.de5(self.de_pd5(h4)))
        return out

    def classify(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        # Step 1: Extract features
        feat_sem, _ = self.extract(x, is_warping=False)

        # Step 2: Decode features
        decoded = self.decode(feat_sem)

        # Step 3: Classify
        out = self.classify(decoded)
        return out
