  require 'nn'


-- imagine we have one network we are interested in, it is called "p1_mlp"
p1_mlp= nn.Sequential(); p1_mlp:add(nn.Linear(5, 2))

-- But we want to push examples towards or away from each other
-- so we make another copy of it called p2_mlp
-- this *shares* the same weights via the set command, but has its own set of temporary gradient storage
-- that's why we create it again (so that the gradients of the pair don't wipe each other)
p2_mlp= p1_mlp:clone('weight', 'bias')

-- we make a parallel table that takes a pair of examples as input. they both go through the same (cloned) mlp
prl = nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)

-- now we define our top level network that takes this parallel table and computes the cosine distance betweem
-- the pair of outputs
mlp= nn.Sequential()
mlp:add(prl)
mlp:add(nn.CosineDistance())


-- lets make two example vectors
x = torch.rand(10,5)
y = torch.rand(10,5)

-- Grad update function..
function gradUpdate(mlp, x, y, learningRate)
    local pred = mlp:forward(x)
    if pred[1]*y < 1 then
        gradCriterion = torch.Tensor({-y})
        mlp:zeroGradParameters()
        mlp:backward(x, gradCriterion)
        mlp:updateParameters(learningRate)
    end
end

-- push the pair x and y together, the distance should get larger..
for i = 1, 500 do
 gradUpdate(mlp, {x, y}, 1, 0.1)
 if ((i%20)==0) then print(mlp:forward({x, y})[1]);end
end

print('---------')

-- pull apart the pair x and y, the distance should get smaller..

for i = 1, 1000 do
 gradUpdate(mlp, {x, y}, -1, 0.1)
 if ((i%20)==0) then print(mlp:forward({x, y})[1]);end
end