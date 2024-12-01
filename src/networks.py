from torch import nn
import torch
from torchvision import transforms
import numpy as np
import cv2 as cv


# BasicCNN - 1 conv layer, 1 linear layer
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(1), # Flatten conv output for linear layers

            nn.LazyLinear(num_classes)
        )
    
    def forward(self, batch_in):
        # For processing batches during training
        batch_out = self.layers(batch_in)
        return batch_out
    
    def predict(self, single_in):
        # For processing single images during inference
        single_in  = single_in.unsqueeze(0)
        pred = self.layers(single_in)
        pred = torch.argmax(pred, dim=1) if self.num_classes > 1 else pred
        return pred.item()

# LeNet (LeNet-5) - 2 conv layers, 3 linear layers
class LeNet(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()

        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(1),

            nn.LazyLinear(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, batch_in):
        # For processing batches during training
        batch_out = self.layers(batch_in)
        return batch_out
    
    def predict(self, single_in):
        # For processing single images during inference
        single_in  = single_in.unsqueeze(0)
        pred = self.layers(single_in)
        pred = torch.argmax(pred, dim=1) if self.num_classes > 1 else pred
        return pred.item()
    

# AlexNet - 5 conv layers, 3 linear layers
class AlexNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.num_classes = num_classes

        self.layers = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained)

        # Modify final classifier layer to have correct number of outputs
        self.layers.classifier[-1] = nn.LazyLinear(num_classes)
    
    def forward(self, batch_in):
        # For processing batches during training
        batch_out = self.layers(batch_in)
        return batch_out
    
    def predict(self, single_in):
        # For processing single images during inference
        single_in  = single_in.unsqueeze(0)
        pred = self.layers(single_in)
        pred = torch.argmax(pred, dim=1) if self.num_classes > 1 else pred
        return pred.item()
    

# VGG16 - 13 conv layers, 3 linear layers
class VGG16(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.num_classes = num_classes

        self.layers = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained)

        # Modify final classifier layer to have correct number of outputs
        self.layers.classifier[-1] = nn.LazyLinear(num_classes)
    
    def forward(self, batch_in):
        # For processing batches during training
        batch_out = self.layers(batch_in)
        return batch_out
    
    def predict(self, single_in):
        # For processing single images during inference
        single_in  = single_in.unsqueeze(0)
        pred = self.layers(single_in)
        pred = torch.argmax(pred, dim=1) if self.num_classes > 1 else pred
        return pred.item()


# Attention inspired by ...
# Analyses chosen patches from grid as well as whole image
class GridAttentionNet(nn.Module):
    def __init__(self, num_classes, analysis_model, attention_model,
                 grid_size=4, num_patches=2):
        super().__init__()

        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_patches = num_patches

        patch_output = 32

        self.attention_net = attention_model(num_classes=grid_size**2)
        self.fullimg_analysis_net = analysis_model(num_classes=patch_output)
        self.patch_analysis_net = analysis_model(num_classes=patch_output)
        self.final_classifier = nn.Linear(in_features=(self.num_patches + 1)*patch_output, out_features=num_classes)

        # Final classification network:
        total_CNNs = num_patches + 1
        self.final_classifier = nn.Linear(total_CNNs*patch_output, self.num_classes)
    
    def forward_single(self, input):
            input = input.unsqueeze(0) # 1x batch

            # Run attention network to get top N patches
            patch_scores = self.attention_net(input)
            patch_scores = patch_scores.squeeze(0)
            top_patches = torch.argsort(patch_scores, descending=True)[:self.num_patches]
            top_patches = top_patches.numpy()

            # Get original input image to extract patches
            input_img = input.squeeze(0).numpy()
            input_img = np.transpose(input_img, (1, 2, 0))
            input_img_w, input_img_h = input_img.shape[1], input_img.shape[0]

            # Iteratively extract and run patches through CNN
            patch_img_w, patch_img_h = input_img_w//self.grid_size, input_img_h//self.grid_size
            patch_outputs = []

            for patch in top_patches:
                # Get 2D coords of patch
                patch_grid_x = patch % self.grid_size
                patch_grid_y = patch // self.grid_size
            
                # Extract patch from image
                patch_img_x = patch_grid_x*patch_img_w
                patch_img_y = patch_grid_y*patch_img_h
                patch_img = input_img[
                    patch_img_y:patch_img_y+patch_img_h,
                    patch_img_x:patch_img_x+patch_img_w]
            
                # Transform patch back to tensor
                patch_input = transforms.ToTensor()(patch_img)
                patch_input = patch_input.unsqueeze(0)

                # Feed patch through patch CNN
                patch_output = self.patch_analysis_net(patch_input)
                patch_outputs.append(patch_output)
            
            # Analyse full image as well
            img_output = self.fullimg_analysis_net(input)
            patch_outputs.append(img_output)
            
            # Final classification
            final_output = self.final_classifier(torch.cat(patch_outputs, dim=1))
            return final_output

    def forward(self, input_batch):
        output_batch = []
        for input in input_batch:
            output = self.forward_single(input)
            output_batch.append(output.squeeze(0))
        return torch.stack(output_batch, dim=0)
    

# Analyses chosen patches of any size from grid
class VariableAttentionNet(nn.Module):
    def __init__(self, num_classes, attention_model, analysis_model, num_patches=2):
        super(VariableAttentionNet, self).__init__()

        self.num_classes = num_classes
        self.num_patches = num_patches

        patch_output = 50

        self.attention_net = nn.Sequential(attention_model(num_classes=num_patches*4),
                                           nn.Sigmoid()) # Need output to be 0-1
        self.fullimg_analysis_net = analysis_model(num_classes=patch_output)
        self.patch_analysis_net = analysis_model(num_classes=patch_output)
        self.final_classifier = nn.Linear(num_patches*patch_output, self.num_classes)
    
    def forward_single(self, input):
            input = input.unsqueeze(0) # 1x batch

            # Run attention network to get top N patches
            attention_output = self.attention_net(input)
            attention_output = attention_output.squeeze(0)
            attention_output = attention_output.detach().numpy()

            # Get original input image to extract patches
            input_img = input.squeeze(0).numpy()
            input_img = np.transpose(input_img, (1, 2, 0))
            input_img_w, input_img_h = input_img.shape[1], input_img.shape[0]

            # Iteratively extract and run patches through CNN
            patch_outputs = []

            for patch_num in range(0, self.num_patches):
                # Get 2D coords of patch
                n = patch_num*4
                patch_x = int(attention_output[n] * input_img_w)
                patch_y = int(attention_output[n+1] * input_img_h)
                patch_w = int(attention_output[n+2] * input_img_w)
                patch_h = int(attention_output[n+2] * input_img_h)

                patch_x = max(1, patch_x)
                patch_y = max(1, patch_y)
                patch_w = max(1, patch_w)
                patch_h = max(1, patch_h)

                # Extract patch from image
                patch_img = input_img[patch_y:patch_y+patch_h,
                            patch_x:patch_x+patch_w]
                patch_img = cv.resize(patch_img, (100, 100))    # Patch cnn needs
                                                                # consistent size
                # Transform patch back to tensor
                patch_input = transforms.ToTensor()(patch_img)
                patch_input = patch_input.unsqueeze(0)

                # Feed patch through patch CNN
                patch_output = self.patch_analysis_net(patch_input)
                patch_outputs.append(patch_output)
            
            # Final classification
            final_output = self.final_classifier(torch.cat(patch_outputs, dim=1))
            return final_output

    def forward(self, input_batch):
        output_batch = []
        for input in input_batch:
            output = self.forward_single(input)
            output_batch.append(output.squeeze(0))
        return torch.stack(output_batch, dim=0)