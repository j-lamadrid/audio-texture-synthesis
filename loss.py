import torch
import torch.nn as nn

class TextureLoss(nn.Module):
    
    def __init__(self, target_activations):
        super(TextureLoss, self).__init__()
        
        self.target_activations = target_activations
    
    
    def H(self, input):
        mat = input.squeeze(0)
        out = torch.sum(mat.unsqueeze(1) * mat.unsqueeze(0), dim=-1)
        return out
    
    
    def ustyu_loss(self, target, noise):
    
        num = torch.norm((target - noise), p=2)
        den = torch.norm(target, p=2)

        return torch.div(num, den)
    
    
    def forward(self, activations):
        
        losses = []
    
        for i in range(len(self.target_activations)):
            act = self.H(activations[i])
            target = self.H(self.target_activations[i])
        
            losses.append((self.ustyu_loss(target, act)).unsqueeze(0))
        
        loss = torch.stack(losses, dim=0).sum(dim=0)
        return loss