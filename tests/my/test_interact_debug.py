import code
import sys
import torch
import torch.optim as optim
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

global_var = 400

class MAIN:
    def __init__(self):
        self.a = 100
        self.b = 200
    def hello(self):
        interact_locals = dict(**globals())
        interact_locals.update(locals())
        code.interact(local=interact_locals)

def main():
    a = 100
    b = 120
    M = MAIN()
    M.hello()
    print("hello")

def test_adam():
    features = torch.tensor([0.01,0.01,0.01], requires_grad=True)
    optimizer = optim.Adam([features], lr=0.01)
    def loss_0(feature):
        return feature[0]**2
    def loss_1(feature):
        return feature[0]**2 + 0.01*feature[1]**2
    def loss_2(feature):
        return feature[0]**2 + feature[1]**2 + 0.01*feature[2]**2
    
    for i in range(10000):
        loss = loss_0(features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(features)
    loss = loss_1(features)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(features)

    for i in range(9999):
        loss = loss_1(features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(features)
    print(optimizer.state)
    print(optimizer.param_groups[0]['eps'])
    loss = loss_2(features)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(features)
    print(optimizer.state)

class MyTri_Backward(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, g):
        inputs = (x, g)
        ctx.save_for_backward(*inputs)
        return g*3*x**2

    @staticmethod
    @custom_bwd
    def backward(ctx, *dont_know_what):
        print(dont_know_what)
        assert 0

mytri_backward = MyTri_Backward.apply

class MyTri(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**3

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        grad = g*3*x**2
        return grad

my_tri = MyTri.apply

def test_second_level_derivative():
    field = torch.tensor([2.], requires_grad=True)
    cell_features = torch.sin(field)

    with torch.no_grad():
        features = cell_features**2
        features.requires_grad = True
        
    weight = torch.tensor([0.1], requires_grad=True)
    mid = features * weight
    rgb = my_tri(mid)
    loss = (rgb - 1)**2

    loss.backward(create_graph=True)

    code.interact(local=locals())
    print(x)
    print(x.grad)

    x_grad_grad = torch.autograd.grad(x.grad, x)

    print(x.grad)
    print(x_grad_grad)

test_second_level_derivative()