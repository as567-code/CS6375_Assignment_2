def create_modules(self):
    modules = nn.Sequential()
    
    # Layer 1
    modules.add_module('Conv1', nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1))
    modules.add_module('ReLU1', nn.ReLU(inplace=True))
    modules.add_module('MaxPool1', nn.MaxPool2d(kernel_size=2, stride=2))
    
    # Layer 2
    modules.add_module('Conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
    modules.add_module('ReLU2', nn.ReLU(inplace=True))
    modules.add_module('MaxPool2', nn.MaxPool2d(kernel_size=2, stride=2))
    
    # Layer 3
    modules.add_module('Conv3', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
    modules.add_module('ReLU3', nn.ReLU(inplace=True))
    modules.add_module('MaxPool3', nn.MaxPool2d(kernel_size=2, stride=2))
    
    # Layer 4
    modules.add_module('Conv4', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
    modules.add_module('ReLU4', nn.ReLU(inplace=True))
    modules.add_module('MaxPool4', nn.MaxPool2d(kernel_size=2, stride=2))
    
    # Layer 5
    modules.add_module('Conv5', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
    modules.add_module('ReLU5', nn.ReLU(inplace=True))
    modules.add_module('MaxPool5', nn.MaxPool2d(kernel_size=2, stride=2))
    
    # Layer 6
    modules.add_module('Conv6', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
    modules.add_module('ReLU6', nn.ReLU(inplace=True))
    modules.add_module('MaxPool6', nn.MaxPool2d(kernel_size=2, stride=2))
    
    # Layer 7
    modules.add_module('Conv7', nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1))
    modules.add_module('ReLU7', nn.ReLU(inplace=True))
    
    # Layer 8
    modules.add_module('Conv8', nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
    modules.add_module('ReLU8', nn.ReLU(inplace=True))
    
    # Layer 9
    modules.add_module('Conv9', nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
    modules.add_module('ReLU9', nn.ReLU(inplace=True))
    
    # Flatten layer
    modules.add_module('Flatten', nn.Flatten())
    
    # Fully connected layers
    modules.add_module('FC1', nn.Linear(50176, 256))
    modules.add_module('FC2', nn.Linear(256, 256))
    
    # Output layer: 7×7×(5B+C) where B=2 boxes, C=1 class
    # So that's 7×7×(5×2+1) = 7×7×11 = 539
    modules.add_module('FC_Output', nn.Linear(256, 7 * 7 * (5 * self.num_boxes + self.num_classes)))
    
    # Sigmoid activation for the output
    modules.add_module('Sigmoid', nn.Sigmoid())
    
    return modules