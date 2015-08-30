
local PairInputStream = require 'PairInputStream'
local Normalizer = require 'Normalizer'
local tds = require 'tds'
require 'nngraph'
require 'nn'
require 'DSSM_MMI_Criterion'
require 'optim'

require 'CDSSM_SparseLinear'
local DSSM_Train = {}
DSSM_Train.__index = DSSM_Train

function DSSM_Train.init()
    self = {}
    setmetatable(self, DSSM_Train)

    self.PairStream = nil
    self.PairValidStream = nil
    self.SrcNorm = nil
    self.TgtNorm = nil
    self.pairTrainFileIdx = 0

    self.model = nil
    return self
end


function DSSM_Train:LoadTrainData(data_dir, qFileName, dFileName, nceProbDisFile, opt)
    -- doing shuffle and ...

    local qData, dData = self:LoadPairDataAtIdx(data_dir, qFileName, dFileName, nceProbDisFile, opt)

    print('Loading training doc query stream done')
    return qData, dData
end
function DSSM_Train:LoadPairDataAtIdx(data_dir, qFileName, dFileName, nceProbDisFile, opt)

    self.PairStream = PairInputStream.init()

    local qData, dData = self.PairStream:Load_Train_PairData(data_dir, qFileName, dFileName, nceProbDisFile, opt)

    self.ScrNorm = Normalizer.CreateFeatureNormalize(opt.Q_FeaNorm, self.PairStream.qStream.feature_size)
    self.TgtNorm = Normalizer.CreateFeatureNormalize(opt.D_FeaNorm, self.PairStream.dStream.feature_size)

    -- PairStream:initFeatureNorm()
    self.pairTrainFileIdx = 0

    return qData, dData
end

function DSSM_Train:ModelInit_FromConfig(opt)

    -- we first try to build fix structure model
    print('Initializing the Model...')
    local Q_feature_size = self.PairStream.qStream.feature_size
    local D_feature_size = self.PairStream.dStream.feature_size

    local wind_size = 1
    local batch_size = 1024
    -- input is feature_size * window_size, filter is the same size. if we use the 
    -- 3D tensor, then the [dw] == 1

    -- Using gModule to construct the network
    -- 3D initilzation is saving the memory than 2D initilzation

    local Q_1_layer = nn.CDSSM_SparseLinear(Q_feature_size * wind_size, 1000)()

    local Q_1_link = nn.Tanh()(Q_1_layer)

    --local Q_1_pool = nn.TemporalMaxPooling(18)(Q_1_link)
    --local Q_1_reshape = nn.Reshape(1000, true)(Q_1_pool)
    -- second layer
    local Q_2_layer = nn.Linear(1000, 300)(Q_1_link)



    local D_1_layer = nn.CDSSM_SparseLinear(D_feature_size * wind_size, 1000)()
    local D_1_link = nn.Tanh()(D_1_layer)
    -- then a max pooling layer
    --local D_1_pool = nn.TemporalMaxPooling(18)(D_1_link)
    --local D_1_reshape = nn.Reshape(1000, true)(D_1_pool)
    -- second layer
    local D_2_layer = nn.Linear(1000, 300)(D_1_link)

    local model = nn.gModule({Q_1_layer, D_1_layer}, {Q_2_layer, D_2_layer})
    -- get the non-sparse query, document and index
    --self.model = model
    for indexNode, node in ipairs(model.forwardnodes) do
    
      if node.data.module then
        print(indexNode, node.data.module)
        
        if indexNode == 4 then
            local weight = torch.load('Q_L1_weight'):view(-1,1000)
            node.data.module.weight = weight:t()
        end
        if indexNode == 6 then
            local weight = torch.load('Q_L2_weight'):view(-1,300)
            node.data.module.weight = weight:t()
            node.data.module.bias:zero()
        end
        if indexNode == 8 then
            local weight = torch.load('D_L1_weight'):view(-1,1000)
            node.data.module.weight = weight:t()
        end

        if indexNode == 10 then
            local weight = torch.load('D_L2_weight'):view(-1,300)
            node.data.module.weight = weight:t()
            node.data.module.bias:zero()
        end     
      end
    end    

    self.model = model
    parameters,gradParameters = self.model:getParameters()

    --print(Q_feature_size)
    --print(parameters:size())
    
    self.optimState = {
      learningRate = opt.learning_rate,
      weightDecay = opt.weight_decay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
    }
    self.optimMethod = optim.sgd
end

function DSSM_Train:Training(qData, dData, opt)
    self.PairStream:Init_Batch()
    local trainingLoss = 0

    for i = 1,self.PairStream.qStream.batch_num do
        --xlua.progress(i, self.PairStream.qStream.batch_num)
        local flag, qData, dData = self.PairStream:Next_Batch(qData, dData, srcNorm, tgtNorm, opt)
        -- doing the forward process
        
        -- negtive samping index
        local batch_size = self.PairStream.qStream.dataFun.batch_size

        local feval = function(x)
                if x ~= parameters then
                    parameters:copy(x)
                end
                gradParameters:zero()
                -- get the cosine similarity for all positive and negtive.
                --local output = self.model:forward({qData, dData})
                --torch.save('output', output)
                local output = torch.load('output')
                -- initialize the criterion function here, in order we can intilized the negative sampling 
                -- differently for each batch

                if opt.objective == 'NCE' then
                -- load the doc distance for NCE Training.

                end

                if opt.objective == 'MMI' then
                    self.criterion = nn.DSSM_MMI_Criterion(batch_size, opt.ntrial, opt.gamma)
                end

                -- forward the criterion
                local err = self.criterion:updateOutput(output)
                print(err)
                local do_df = self.criterion:updateGradInput()
                self.model:backward(output, do_df)

                return err, gradParameters
            end
            self.optimMethod(feval, parameters, self.optimState)
    end
end

return DSSM_Train