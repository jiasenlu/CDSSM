-- Preprocess the code to generate the Sparse input format.


require 'torch'

local batch_size = 1024
local WordHash = require 'WordHash'
local ComputelogPD = require 'ComputelogPD'
local th_utils = require 'th_utils'

-- generate the vocabulary, the answer should not do any processing.
local inFile = 'data/train.pair.tok.tsv'
local outFile = 'data/train'

if not th_utils.IsFile(outFile..'.src.l3g.txt') then
	print('Creating Voc file for '.. inFile)
	WordHash.Pair2Voc(inFile, 3, 1, 2, outFile, 0)
end

-- creating pair to sequence feature.
local inFile = 'data/train.pair.tok.tsv'
local srcVocFile = 'data/train.src.l3g.txt'
local tgtVocFile = 'data/train.tgt.l3g.txt'
local nMaxLength = 20
local outFile = 'data/train'

if not (th_utils.IsFile(outFile..'.src.seq.fea') and th_utils.IsFile(outFile..'.tgt.seq.fea') 
				and th_utils.IsFile(outFile..'.NegTgt.seq.fea')) then
	print('Creating Pair2Seq Feature')
	WordHash.Pair2SeqFea(inFile, srcVocFile, tgtVocFile, nMaxLength, 1, 2, outFile, 'l3g')
end


local inFile = 'data/train.src.seq.fea'
local BatchSize = batch_size
local outFile = 'data/train.src.seq.sparse.t7'
local WinSize = 3

--if not th_utils.IsFile(outFile) then
	print('Converting Seq feature to bin: '.. inFile .. ' BatchSize: ' .. BatchSize)
	WordHash.SeqFea2SparseMatrix(inFile, BatchSize, WinSize, 2118, outFile)
--end

local inFile = 'data/train.tgt.seq.fea'
local BatchSize = batch_size
local outFile = 'data/train.tgt.seq.sparse.t7'
local WinSize = 3
if not th_utils.IsFile(outFile) then
	print('Converting Seq feature to bin: '.. inFile.. ' BatchSize: ' .. BatchSize)	
	WordHash.SeqFea2SparseMatrix(inFile, BatchSize, WinSize, 2255, outFile)
end
--[[
-- generate validation validation file
local inFile = 'data/vqa/VQA_pair_val.txt'
local srcVocFile = 'data/vqa/train.src.l3g.txt'
local tgtVocFile = 'data/vqa/train.tgt.l3g.txt'
local nMaxLength = 1
local outFile = 'data/vqa/val'

if not (th_utils.IsFile(outFile..'.src.seq.fea') and th_utils.IsFile(outFile..'.tgt.seq.fea') 
				and th_utils.IsFile(outFile..'.NegTgt.seq.fea')) then
	print('Creating Pair2Seq Feature for '..inFile)
	WordHash.Pair2SeqFea(inFile, srcVocFile, tgtVocFile, nMaxLength, 1, 2, outFile, 'l3g')
end

local inFile = 'data/vqa/val.src.seq.fea'
local BatchSize = batch_size
local outFile = 'data/vqa/val.src.seq.fea.t7'

if not th_utils.IsFile(outFile) then
	print('Converting Seq feature to bin: '.. inFile .. ' BatchSize: ' .. BatchSize)
	WordHash.SeqFea2Bin(inFile, BatchSize, outFile)
end

local inFile = 'data/vqa/val.tgt.seq.fea'
local BatchSize = batch_size
local outFile = 'data/vqa/val.tgt.seq.fea.t7'

if not th_utils.IsFile(outFile) then
	print('Converting Seq feature to bin: '.. inFile.. ' BatchSize: ' .. BatchSize)	
	WordHash.SeqFea2Bin(inFile, BatchSize, outFile)
end

-- generate test file (the test file don't have GT, and is for prediction)

local inFile = 'data/vqa/VQA_pair_test.txt'
local srcVocFile = 'data/vqa/train.src.l3g.txt'
local tgtVocFile = 'data/vqa/train.tgt.l3g.txt'
local nMaxLength = 1
local outFile = 'data/vqa/test'

if not (th_utils.IsFile(outFile..'.src.seq.fea') and th_utils.IsFile(outFile..'.tgt.seq.fea') 
				and th_utils.IsFile(outFile..'.NegTgt.seq.fea')) then
	print('Creating Pair2Seq Feature for '..inFile)
	WordHash.Pair2SeqFea(inFile, srcVocFile, tgtVocFile, nMaxLength, 1, 2, outFile, 'l3g')
end

local inFile = 'data/vqa/test.src.seq.fea'
local BatchSize = batch_size
local outFile = 'data/vqa/test.src.seq.fea.t7'

if not th_utils.IsFile(outFile) then
	print('Converting Seq feature to bin: '.. inFile .. ' BatchSize: ' .. BatchSize)
	WordHash.SeqFea2Bin(inFile, BatchSize, outFile)
end

local inFile = 'data/vqa/test.tgt.seq.fea'
local BatchSize = batch_size
local outFile = 'data/vqa/test.tgt.seq.fea.t7'

if not th_utils.IsFile(outFile) then
	print('Converting Seq feature to bin: '.. inFile.. ' BatchSize: ' .. BatchSize)	
	WordHash.SeqFea2Bin(inFile, BatchSize, outFile)
end
]]--