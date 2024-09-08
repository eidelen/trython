# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py

from __future__ import print_function
import torch

print(torch.__version__)

x = torch.rand(3,4)
print(x)

x = x.new_ones(5,4);
print(x)

print(x.size())

x.fill_(10)
print(x)

x = torch.rand_like(x)
print(x)


a = torch.ones(  (3,4) ) * 2
b = torch.ones( (3,4) )
r = torch.empty( (3,4) )
torch.add( a, b, out=r)
print( r )

print( 'Cuda is' + (' on' if torch.cuda.is_available() else ' off'))
