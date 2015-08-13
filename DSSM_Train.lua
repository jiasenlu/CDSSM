
local PairInputStream = require 'PairInputStream'
local Normalizer = require 'Normalizer'

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






return DSSM_Train