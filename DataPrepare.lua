--[[ 
Data preparation before intilizaed the model. Including:

Construct ShuffleTrainFiles if multiple input ?? need more info here.

Get the dimension of document and query, get the normalizer, NCE etc.

]]--
local PairInputStream = require 'PairInputStream'

local PairInputStream = {}
PairInputStream.__index = PairInputStream

function PairInputStream.init()
    local self = {}
    setmetatable(self, PairInputStream)

    self.Query_MaxSegment_batch = 40000
    self.Doc_MaxSegment_batch = 40000
    self.Doc_MaxSegment_batch = 40000

    self.srNCEProbDist = nil
    self.qStream = nil
    self.dStream = nil

end


function PairInputStream:Load_Train_PairData(data_dir, qFileName, dFileName, nceProbDisFile, opt)
    
    PairInputStream:Load_PairData(data_dir, qFileName, dFileName, nceProbDisFile)

    if opt.feature_dimension_query <= 0 or opt.feature_dimension_doc <= 0 then
        opt.feature_dimension_query = qStream.feature_size
        opt.feature_dimension_doc = qStream.feature_size
    end

    if opt.mirror_init == 1 then
        -- need to implement that.
    end

end


function PairInputStream:Load_PairData(data_dir, qFileName, dFileName, nceProbDisFile)
    self.qStream = SequenceInputStream.init()
    self.qStream:get_dimension(data_dir, fileName, opt)

    self.dStream = SequenceInputStream.init()
    self.dStream:get_dimension(data_dir, fileName, opt)

    if nceProbDisFile ~= nil then

    end

    self.Query_MaxSegment_batch = math.max(self.Query_MaxSegment_batch, qStream.maxSequence_perBatch)
    self.Doc_MaxSegment_batch = math.max(self.Doc_MaxSegment_batch, dStream.maxSequence_perBatch)
    self.maxSegment_batch = math.max(self.Query_MaxSegment_batch, Doc_MaxSegment_batch)

end



return DataPrepare