
local PairInputStream = require 'PairInputStream'
local Normalizer = require 'Normalizer'
local tds = require 'tds'
require 'nngraph'
require 'nn'
require 'DSSM_MMI_Criterion'
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

    local Q_feature_size = self.PairStream.qStream.feature_size
    local D_feature_size = self.PairStream.dStream.feature_size

    local wind_size = 3
    local batch_size = 1024
    -- input is feature_size * window_size, filter is the same size. if we use the 
    -- 3D tensor, then the [dw] == 1

    -- Using gModule to construct the network
    -- 3D initilzation is saving the memory than 2D initilzation
    local Q_1_layer = nn.TemporalConvolution(Q_feature_size * wind_size, 1000, 1)()
    local Q_1_link = nn.Tanh()(Q_1_layer)
    -- then a max pooling layer
    local Q_1_pool = nn.TemporalMaxPooling(18)(Q_1_link)
    local Q_1_reshape = nn.Reshape(1000, true)(Q_1_pool)
    -- second layer
    local Q_2_layer = nn.Linear(1000, 300)(Q_1_reshape)
    local Q_2_link = nn.Tanh()(Q_2_layer)

    local D_1_layer = nn.TemporalConvolution(D_feature_size * wind_size, 1000, 1)()
    local D_1_link = nn.Tanh()(D_1_layer)
    -- then a max pooling layer
    local D_1_pool = nn.TemporalMaxPooling(18)(D_1_link)
    local D_1_reshape = nn.Reshape(1000, true)(D_1_pool)
    -- second layer
    local D_2_layer = nn.Linear(1000, 300)(D_1_reshape)
    local D_2_link = nn.Tanh()(D_2_layer)

    local model = nn.gModule({Q_1_layer, D_1_layer}, {Q_2_link, D_2_link})

    -- get the non-sparse query, document and index
    self.model = model


end

function DSSM_Train:Training(qData, dData, opt)
    self.PairStream:Init_Batch()
    local trainingLoss = 0

    for i = 1,self.PairStream.qStream.batch_num do
        local flag, qData, dData = self.PairStream:Next_Batch(qData, dData, srcNorm, tgtNorm, opt)
        -- doing the forward process
        
        -- negtive samping index
        local batch_size = self.PairStream.qStream.dataFun.batch_size

        local qTensor = qData.data_matrix:double()
        local dTensor = dData.data_matrix:double()

        -- get the cosine similarity for all positive and negtive.
        local output = self.model:forward({qTensor, dTensor})
        
        -- initialize the criterion function here, in order we can intilized the negative sampling 
        -- differently for each batch

        if opt.objective == 'NCE' then
        -- load the doc distance for NCE Training.

        end

        if opt.objective == 'MMI' then
            self.criterion = nn.DSSM_MMI_Criterion(batch_size, opt.ntrial)
        end


        -- forward the criterion
        local alpha = self.criterion:updateOutput(output)
        print(alpha)
        error()
    end
end

return DSSM_Train