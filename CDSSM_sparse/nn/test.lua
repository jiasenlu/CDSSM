
require 'nn'


a = nn.SparseLinearBatch(2, 3)

data = torch.Tensor({{1,1,1}, {1,2,1}, {2,1,1}})
out = a:forward(data)
grad = torch.Tensor(2,3):fill(1)
--a.weight:fill(1)
print(a:backward(data, grad))
print(a.weight)
print(a.bias)

data = torch.Tensor({{1,1},{1,0}})
b = nn.Linear(2,3)
b.weight:copy(a.weight)
b.bias:copy(a.bias)
out = b:forward(data)
grad = torch.Tensor(2,3):fill(1)

--print(b.weight)
print(b:backward(data, grad))
--print(b.gradWeight)
print(b.bias)
print(b.weight)


c = nn.CDSSMSparseConvolution(2,3,10)
--c = nn.SparseLinearBatch(2, 3)
data = torch.Tensor({{1,1,1,1}, {1,1,2,1},{2,1,1,1}})

--c.weight:fill(1)
--c.bias:fill(0)
c.weight:copy(a.weight)
c.bias:copy(a.bias)

print(c:forward(data))

grad = torch.Tensor(2,10,3):fill(1)

print(c:backward(data, grad))
print(c.weight)
print(c.bias)
--print(data)