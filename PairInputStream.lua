--[[ 
Data preparation before intilizaed the model. Including:

Construct ShuffleTrainFiles if multiple input ?? need more info here.

Get the dimension of document and query, get the normalizer, NCE etc.

]]--
local SequenceInputStream = require 'SequenceInputStream'

local PairInputStream = {}
PairInputStream.__index = PairInputStream

function PairInputStream.init()
    local self = {}
    setmetatable(self, PairInputStream)

    self.Query_MaxSegment_batch = 40000
    self.Doc_MaxSegment_batch = 40000
    self.maxSegment_batch = 40000
    self.srNCEProbDist = nil
    self.qStream = nil
    self.dStream = nil

    return self
end


function PairInputStream:Load_Train_PairData(data_dir, qFileName, dFileName, nceProbDisFile, opt)
    
    self:Load_PairData(data_dir, qFileName, dFileName, nceProbDisFile)

    if opt.feature_dimension_query <= 0 or opt.feature_dimension_doc <= 0 then
        opt.feature_dimension_query = self.qStream.feature_size
        opt.feature_dimension_doc = self.qStream.feature_size
    end

    if opt.mirror_init == 1 then
        -- need to implement that.
    end
end


function PairInputStream:Load_PairData(data_dir, qFileName, dFileName, nceProbDisFile)
    self.qStream = SequenceInputStream.init()
    self.qStream:get_dimension(data_dir, qFileName, opt)

    self.dStream = SequenceInputStream.init()
    self.dStream:get_dimension(data_dir, dFileName, opt)

    if nceProbDisFile ~= nil then

    end
    self.Query_MaxSegment_batch = math.max(self.Query_MaxSegment_batch, self.qStream.maxSequence_perBatch)
    self.Doc_MaxSegment_batch = math.max(self.Doc_MaxSegment_batch, self.dStream.maxSequence_perBatch)
    self.maxSegment_batch = math.max(self.Query_MaxSegment_batch, self.Doc_MaxSegment_batch)

end

function PairInputStream:InitFeatureNorm(srcNormalizer, tgtNormalizer)
    -- only handle min_max normalization here.

end

function PairInputStream:Init_Batch()
    -- set the batch index to be 0
    self.qStream:init_batch()
    self.dStream:init_batch()

end

function PairInputStream:Next_Batch(srcNorm, tgtNorm, opt)
    -- get the next batch data.
    if (not self.qStream:Fill(opt.feature_dimension_query)) or (not self.dStream:Fill(opt.feature_dimension_doc)) then
        return false
    end




end
return PairInputStream