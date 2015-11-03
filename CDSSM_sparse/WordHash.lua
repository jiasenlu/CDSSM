--[[ 
Implementation of CDSSM Model in Lua. The function name are the same
of the orginial C# implementation. 

]]--
--local iconv = require "iconv"
local utils = require 'utils'
local Batch = require 'Batch'
local th_utils = require 'th_utils'
local WordHash = {}
WordHash.__index = WordHash
function WordHash.GenLable(inFile, uniqueFile, outFile)
    
    local label = {}
    local unique = {}
    local f = assert(io.open(uniqueFile, "r"))
    
    for line in f:lines() do
        table.insert(unique, line)
    end
    count = 0
    local f = assert(io.open(inFile, "r"))
    for line in f:lines() do 
        count = count + 1
        -- split the src and tgt by '\t'
        local sent = {}
        for value in line:gmatch("[^\t]+") do 
            table.insert(sent, value)
        end

        for i = 1, #unique do
            if sent[2] == unique[i] then
                label[count] = i
            end
        end
    end

    torch.save('label.t7', label)

end

function WordHash.Pair2Voc(inFile, N, srcIdx, tgtIdx, outFile, STsame)
    -- get the LNg vocabulary for src and tgt. if STsame == 1, then src and tgt will get same vocabulary, 
    -- otherwise, different vocab file

    local srcVoc, tgtVoc
    srcVoc = {}
    tgtVoc = {}

    --local decoder = iconv.new('utf8','utf8')
    local f = assert(io.open(inFile, "r"))

    for line in f:lines() do     
        -- split the src and tgt by '\t'
        --line = decoder:iconv(line)
        local sent = {}
        for value in line:gmatch("[^\t]+") do             
            table.insert(sent, value)
        end
        -- get the seq l3g for each sentence.
        local featStrFeq = utils.String2FeatStrSeq(sent[srcIdx], N, 1, 'l3g')
        -- get the seq l3g for each sentence.
        local featTgtFeq = utils.String2FeatStrSeq(sent[tgtIdx], N, 1, 'l3g')
        if STsame == 1 then
            for key, value in pairs(featStrFeq[1]) do
                if not srcVoc[key] then
                    srcVoc[key] = 1
                else
                    srcVoc[key] = srcVoc[key] + value
                end
            end
            for key, value in pairs(featTgtFeq[1]) do
                if not srcVoc[key] then
                    srcVoc[key] = 1

                else
                    srcVoc[key] = srcVoc[key] + value
                end
            end
        else
            for key, value in pairs(featStrFeq[1]) do
                if not srcVoc[key] then
                    srcVoc[key] = 1
                else
                    srcVoc[key] = srcVoc[key] + value
                end
            end            
        
            for key, value in pairs(featTgtFeq[1]) do
                if not tgtVoc[key] then
                    tgtVoc[key] = 1
                else
                    tgtVoc[key] = tgtVoc[key] + value
                end
            end
        end
    end
    f:close()

    if STsame == 1 then
        tgtVoc = srcVoc
    end
    -- ranked by frequency
    srcVoc = th_utils.spairs(srcVoc, function(t,a,b) return t[b]<t[a] end)
    tgtVoc = th_utils.spairs(tgtVoc, function(t,a,b) return t[b]<t[a] end)

    print('srcVoc contains vocabulary: '..#srcVoc)
    print('tgtVoc contains vocabulary: '..#tgtVoc)
    local outSrcVocFile = outFile .. '.src.l' .. N .. 'g.txt'
    local outTgtVocFile = outFile .. '.tgt.l' .. N .. 'g.txt'
    local fSrc = assert(io.open(outSrcVocFile, "w+"))

    for key, value in pairs(srcVoc) do
        fSrc:write(value, '\n')
    end
    fSrc:close()
    local fTgt = assert(io.open(outTgtVocFile, "w+"))    
    for key, value in pairs(tgtVoc) do
        fTgt:write(value, '\n') 
    end
    fTgt:close()


end

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
    --local decoder = iconv.new('utf8','utf8')

    local f = assert(io.open(inFile, "r"))
    for line in f:lines() do 
        --line = decoder:iconv(line)
        
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
function WordHash.SeqFea2SparseMatrix(inFile, BatchSize, WinSize, vocSize, outFile)
    

    if WinSize == 1 then
        local sparseMatrix = {}
        local S = {}
        local nLine = 1
        local f = assert(io.open(inFile, "r"))

        for line in f:lines() do
            if nLine == BatchSize+1 then
                -- create local matrix based on S.
                table.insert(sparseMatrix, torch.IntTensor(S))
                nLine = 1
                S = {}
            end
            local rgWfs = utils.String2Matrix(line)
            for key, val in pairs(rgWfs[1]) do
                table.insert(S, {nLine, key, val})
            end
            nLine = nLine + 1
        end


        -- insert the last one.
        print('saving ' .. outFile)
        table.insert(sparseMatrix, torch.IntTensor(S))
        torch.save(outFile, sparseMatrix)

    else
        -- contain the window size.
        local f = assert(io.open(inFile, "r"))
        local win = torch.floor(WinSize / 2)
        local S = {}
        local sparseMatrix = {}        
        local nLine = 1

        for line in f:lines() do
            if nLine == BatchSize+1 then
                -- create local matrix based on S.
                table.insert(sparseMatrix, torch.IntTensor(S))
                nLine = 1
                S = {}
            end
            local rgWfs = utils.String2Matrix(line)
            local count = 1
            for i = 1,#rgWfs do
                if i-win > 0 and i + win < #rgWfs+1 then
                    for idx = i-win, i+win do
                        for key, val in pairs(rgWfs[idx]) do
                            table.insert(S, {nLine, count, key + vocSize*(idx-i+1), val})
                        end                        
                    end
                    count = count+1              
                end
            end 
            nLine = nLine + 1
        end
        -- insert the last one.
        table.insert(sparseMatrix, torch.IntTensor(S))
        print('saving ' .. outFile)
        torch.save(outFile, sparseMatrix)

    end

end

return WordHash
