import torch
class NeauralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

            #1st Hidden Layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            #2nd Hidden Layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            #Output Layer
            torch.nn.Linear(20, num_outputs),
        )
    def forward(self, x):
        logits = self.layers(x)
        return logits
    
model = NeauralNetwork (50, 3)

print(model)

num_params =sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Number of Parameters:", num_params)

print(model.layers[0].weight)
print(model.layers[0].weight.shape)

print(model.layers[0].bias)
print(model.layers[0].bias.shape)
