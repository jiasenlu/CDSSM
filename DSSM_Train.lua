
local PairInputStream = require 'PairInputStream'
local Normalizer = require 'Normalizer'
require 'nngraph'
require 'nn'
require 'DSSM_CosineDist'


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

    return self
end


function DSSM_Train:LoadTrainData(data_dir, qFileName, dFileName, nceProbDisFile, opt)
    -- doing shuffle and ...

    self:LoadPairDataAtIdx(data_dir, qFileName, dFileName, nceProbDisFile, opt)

    print('Loading training doc query stream done')

end
function DSSM_Train:LoadPairDataAtIdx(data_dir, qFileName, dFileName, nceProbDisFile, opt)

    self.PairStream = PairInputStream.init()

    self.PairStream:Load_Train_PairData(data_dir, qFileName, dFileName, nceProbDisFile, opt)

    self.ScrNorm = Normalizer.CreateFeatureNormalize(opt.Q_FeaNorm, self.PairStream.qStream.feature_size)
    self.TgtNorm = Normalizer.CreateFeatureNormalize(opt.D_FeaNorm, self.PairStream.dStream.feature_size)

    -- PairStream:initFeatureNorm()
    self.pairTrainFileIdx = 0


end

function DSSM_Train:ModelInit_FromConfig(opt)


    -- we first try to build fix structure model

    local Q_feature_size = self.PairStream.qStream.feature_size
    local D_feature_size = self.PairStream.dStream.feature_size

    local wind_size = 3
    local batch_size = 1024
    -- input is feature_size * window_size, filter is the same size. if we use the 
    -- 3D tensor, then the [dw] == 1
    local Index_layer = nn.Identity()()
    local Q_1_layer = nn.TemporalConvolution(Q_feature_size * wind_size, 1000, 1)()
    local Q_1_link = nn.Tanh()(Q_1_layer)
    -- then a max pooling layer
    local Q_1_pool = nn.TemporalMaxPooling(20)(Q_1_link)
    local Q_1_reshape = nn.Reshape(batch_size, 1000)(Q_1_pool)
    -- second layer
    local Q_2_layer = nn.Linear(1000, 300)(Q_1_reshape)
    local Q_2_link = nn.Tanh()(Q_2_layer)

    local D_1_layer = nn.TemporalConvolution(Q_feature_size * wind_size, 1000, 1)()
    local D_1_link = nn.Tanh()(D_1_layer)
    -- then a max pooling layer
    local D_1_pool = nn.TemporalMaxPooling(20)(D_1_link)
    local D_1_reshape = nn.Reshape(batch_size, 1000)(D_1_pool)
    -- second layer
    local D_2_layer = nn.Linear(1000, 300)(D_1_reshape)
    local D_2_link = nn.Tanh()(D_2_layer)

    local Cos_layer = nn.DSSM_CosineDist()({Q_2_link, D_2_link, Index_layer})

    -- add another layer update the 
    local model = nn.gModule({Q_1_layer, D_1_layer, Index_layer}, {Cos_layer})

    -- get the non-sparse query, document and index


    

    
end





return DSSM_Train