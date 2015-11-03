-- Sparse Format CDSSM, Parameter setting

require 'torch'
require 'nn'
local tds = require 'tds'

require 'nngraph'
require 'DSSM_MMI_Criterion'
require 'optim'
local th_utils = require 'th_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train CDSSM model')

cmd:text()
cmd:option('-objective', 'MMI', 'Currently only working NCE and MMI')
cmd:option('-batch_size', 1024, '')
cmd:option('-ntrial', 50, '')
cmd:option('-max_iter', 60, '')
cmd:option('-gamma', 25, '')
cmd:option('-learning_rate', 0.01, '')
cmd:option('-weight_decay',0,'')
cmd:option('-momentum',0.9,'')
cmd:option('-lr', 0.001, '')
cmd:option('-lr_batch', 20, '')
cmd:option('-optim', 'sgd')

cmd:option('-loadmodel', '', '')
cmd:option('-feature_dimension_query', 2118, '')
cmd:option('-feature_dimension_doc', 2255, '')
cmd:option('-mode', 'cpu', '1:gpu, 0:cpu')
cmd:option('-gpu_device', 2, '')

cmd:option('-convo', 1)
cmd:text()

opt = cmd:parse(arg)

if opt.mode == 'gpu' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
   cutorch.setDevice(opt.gpu_device)
end

local data_dir = 'data'
local qFileNameTrain = 'data/train.src.seq.sparse.t7'
local dFileNameTrain = 'data/train.tgt.seq.sparse.t7'

qDataTrain = torch.load(qFileNameTrain)
dDataTrain = torch.load(dFileNameTrain)


-- define the model
local wind_size = 3

local Q_feature_size = opt.feature_dimension_query
local D_feature_size = opt.feature_dimension_doc
if opt.convo == 0 then
    local Q_layer = nn.Sequential()
    Q_layer:add(nn.SparseLinearBatch(Q_feature_size, 1000))
    Q_layer:add(nn.Tanh())
    Q_layer:add(nn.Dropout(0.5))
    Q_layer:add(nn.Linear(1000, 300))
    Q_layer:add(nn.Tanh())


    local D_layer = nn.Sequential()
    D_layer:add(nn.SparseLinearBatch(D_feature_size, 1000))
    D_layer:add(nn.Tanh())
    D_layer:add(nn.Dropout(0.5))        
    D_layer:add(nn.Linear(1000, 300))
    D_layer:add(nn.Tanh())

    model = nn.ParallelTable()
    model:add(Q_layer)
    model:add(D_layer)
else
    local Q_layer = nn.Sequential()
    Q_layer:add(nn.CDSSMSparseConvolution(Q_feature_size * wind_size, 1000, 20))
    Q_layer:add(nn.Tanh())
    Q_layer:add(nn.TemporalMaxPooling(20))
    Q_layer:add(nn.Reshape(1000, true))
    Q_layer:add(nn.Linear(1000, 300))
    
    local D_layer = nn.Sequential()
    D_layer:add(nn.CDSSMSparseConvolution(D_feature_size * wind_size, 1000, 20))
    D_layer:add(nn.Tanh())
    D_layer:add(nn.TemporalMaxPooling(20))
    D_layer:add(nn.Reshape(1000, true))
    D_layer:add(nn.Linear(1000, 300))
    
    model = nn.ParallelTable()
    model:add(Q_layer)
    model:add(D_layer)
end

if opt.mode == 'gpu' then
    model:cuda()
end
 
parameters,gradParameters = model:getParameters()

if opt.optim == 'sgd' then
    optimState = {
      learningRate = opt.learning_rate,
      weightDecay = opt.weight_decay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
    }
    optimMethod = optim.sgd
end
if opt.optim == 'rmsprop' then
    optimState = {
      learningRate = nil,
      alpha = nil,
      epsilon = nil
    }
    optimMethod = optim.rmsprop        
end

print(model)
-- training:

function train(qData, dData, opt, epoch)
    if epoch > opt.lr_batch then
        self.optimState.learningRate = opt.lr
    end
    --print('learning rate: '..optimState.learningRate)
    if opt.loadmodel ~= '' then
        model = torch.load(opt.loadmodel)
    end
    model:training()
    local trainingLoss = 0
    local total_err = 0

    for i = 1, #qData do
        local qTensor, dTensor
        if opt.mode == 'gpu' then
            qTensor = qData[i]:cuda()
            dTensor = dData[i]:cuda()   
        else
            qTensor = qData[i]:double()
            dTensor = dData[i]:double()     
        end      
        batch_size = torch.max(qTensor:select(2,1))
        local feval = function(x)
                if x ~= parameters then
                    parameters:copy(x)
                end         
                gradParameters:zero()
                -- get the cosine similarity for all positive and negtive.
                local output = model:forward({qTensor, dTensor})

                if opt.objective == 'NCE' then
                -- load the doc distance for NCE Training.
                end
                if opt.objective == 'MMI' then
                    criterion = nn.DSSM_MMI_Criterion(batch_size, opt.ntrial, opt.gamma, opt.batch_size)
                end
                if opt.mode == 'gpu' then
                    criterion:cuda()
                end
                local err = criterion:updateOutput(output)
                local do_df = criterion:updateGradInput()
                model:backward({qTensor, dTensor}, do_df)
                gradParameters:div(batch_size)
                total_err = total_err + err
                err = err / batch_size
                return err, gradParameters
            end
        optimMethod(feval, parameters, optimState)

    end
    collectgarbage()
    return total_err
end

for i = 1,opt.max_iter do
    local timer = torch.Timer();    
    total_err = train(qDataTrain, dDataTrain, opt, 1)
    print('training error is: ' .. total_err)
    --print(string.format('load x:%f',timer:time().real));
end
