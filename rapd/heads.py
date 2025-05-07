import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
