import torch
import torch.nn as nn

def backwardhook(module,grad_in,grad_out):
    print("module is",module)
    print("in grad is",grad_in)
    print("out grad is",grad_out)
    grad_out = torch.randn(grad_in)
    print("new grad out",grad_out)

# def reghook()

# def forwardhook(model)

def forwardhook(module,inputtensor,outputtensor):
    print("This is forward process",module)
    print("input tensor is ",inputtensor)
    print("output tensor is",outputtensor)
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4,2),
            nn.ReLU(),
            nn.Linear(2,4)
        )
        self.linear = nn.Linear(4,1,bias=True)
        self.layer.register_backward_hook(backwardhook)
        for param in self.parameters():
            print(param)
        # self.linear.register_forward_hook(forwardhook)
        # self.linear.register_backward_hook(backwardhook)
    def forward(self,x):
        return self.linear(self.layer(x))

if __name__ == "__main__":
    net = Net()
    # for parameter in net.parameters():
    #     print(parameter)
    # tensor = torch.rand((2,4))
    # result = net(tensor)
    # result = torch.mean(result)
    # print("result is",result)
    # result.backward()

# grad_list = []
# def print_grad(grad):
# print("grad is",grad)
# grad = torch.rand_like(grad)
# print("new grad is",grad)
# # grad_list.append(grad)
# return grad
# from torch.autograd import Variable
# x = Variable(torch.randn(2,1),requires_grad=True)
# y = x 
# z = torch.mean(torch.pow(y,2))
# # y.register_hook(print_grad)
# x.register_hook(print_grad)
# z.backward()
# print("x is,",x)
# print("grad of x is",x.grad)
# x = x - x.grad
# print("x new is",x)
# print("x.grad is",x.grad)
# print(y.grad)