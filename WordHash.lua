--[[ 
Implementation of CDSSM Model in Lua. The function name are the same
of the orginial C# implementation. 

]]--

local utils = require 'utils'
local Batch = require 'Batch'

local WordHash = {}
WordHash.__index = WordHash

function WordHash.Pair2SeqFea(inFile, srcVocFile, tgtVocFile, nMaxLength, srcIdx, tgtIdx, outFile, featType)

-- here the input is the pair. 
    local N = 3
    
    -- load the source vocabulary file
    local f = torch.DiskFile(srcVocFile)
    local rawdata = f:readString('*a')
    f:close()
    local srcVoc = {}
    local count = 1
    for key in rawdata:gmatch("[^%s]+") do 
        srcVoc[key] = count
        count = count + 1
    end
    
    -- load the target vocabulary file
    local f = torch.DiskFile(tgtVocFile)
    local rawdata = f:readString('*a')
    f:close()
    local tgtVoc = {}
    local count = 1
    for key in rawdata:gmatch("[^%s]+") do 
        tgtVoc[key] = count 
        count = count + 1
    end

    -- load the inFile data.
    local outSrcFile = outFile .. '.src.seq.fea'
    local outTgtFile = outFile .. '.tgt.seq.fea'

    local fSrc = assert(io.open(outSrcFile, "w+"))
    local fTgt = assert(io.open(outTgtFile, "w+"))

    local f = assert(io.open(inFile, "r"))
    for line in f:lines() do 
        
        -- split the src and tgt by '\t'
        local sent = {}
        for value in line:gmatch("[^\t]+") do 
            table.insert(sent, value)
        end

        -- get the seq l3g for each sentence.
        local featStrFeq = utils.String2FeatStrSeq(sent[srcIdx], N, nMaxLength, featType)
        -- get the index based on vocabulary mapping
        local rgSrcWfs = utils.StrFreq2IdFreq(featStrFeq, srcVoc)

        -- get the seq l3g for each sentence.
        local featTgtFeq = utils.String2FeatStrSeq(sent[tgtIdx], N, nMaxLength, featType)
        -- get the index based on vocabulary mapping
        local rgTgtWfs = utils.StrFreq2IdFreq(featTgtFeq, tgtVoc)

        -- convert matrix to a string of key:value wih # as separator btw vector
        local out1 = utils.Matrix2String(rgSrcWfs)
        local out2 = utils.Matrix2String(rgTgtWfs)

        -- write to txt file in order to maintain the consistency with orginal code and make 
        -- the code smaller.

        fSrc:write(out1, '\n')
        fTgt:write(out2, '\n')
    end
    f:close()
    fSrc:close()
    fTgt:close()
end
-- for the l3g dictionary, we can either provide the l3g.txt, 
-- or generate from the train txt. 


function WordHash.SeqFea2Bin(inFile, BatchSize, outFile)
-- batch
-- BatchSize
-- ElementSize
-- FeatureDim
-- m_rgSampleIdx: accumulate count of the size
-- m_rgSegIdx: segmentation count
-- m_rgFeaIdx: feat index
-- m_rgFeaVal: feat number
-- nMaxFeatureDimension: maxDimension.
    local nMaxFeatureDimension = 0
    local nMaxFeatureNumPerBatch = 0
    local featureDimension = 0
    local nMaxSegmentSize = 0

    local batch = Batch.init()

    local f = assert(io.open(inFile, "r"))
    for line in f:lines() do 
        
        local rgWfs = utils.String2Matrix(line)
        --print(mt)
        
        if batch.BatchSize == BatchSize then


        end
        
        local featureDimension = batch:LoadSeqSample(rgWfs)
        if featureDimension > nMaxFeatureDimension then
            nMaxFeatureDimension = featureDimension
        end
        if batch.SegSize() > nMaxSegmentSize then
            nMaxSegmentSize = batch.SegSize()
        end
        print(nMaxFeatureDimension)
        print(nMaxSegmentSize)
        break
    end

    if batch.BatchSize() > 0 then
        if batch.ElementSize() > nMaxFeatureNumPerBatch then
            nMaxFeatureNumPerBatch = batch.ElementSize()
        end
        batch.WriteSeqSample()
        batch.Clear()
    end

end



return WordHash