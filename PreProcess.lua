-- Test the function. 

require 'torch'

local batch_size = 512
local WordHash = require 'WordHash'
local ComputelogPD = require 'ComputelogPD'
local SequenceInputStream = require 'SequenceInputStream'
local th_utils = require 'th_utils'

-- generate the vocabulary, the answer should not do any processing.
local inFile = 'data/vqa/VQA_pair_train.txt'
local outFile = 'data/vqa/train'

if not th_utils.IsFile(outFile..'.src.l3g.txt') then
	print('Creating Voc file for '.. inFile)
	WordHash.Pair2Voc(inFile, 3, 1, 2, outFile, 0)
end

-- during testing, for each batch, we first generate all unique label in training, and 
-- for each batch, we rank the answers by all the unqiue label. (for open-ended)

-- for each questions in testing, using the multiple choices, and rank to get the predict 
-- labels. (for multiple choice.)

local inFile = 'data/vqa/VQA_pair_train.txt'
local srcVocFile = 'data/vqa/train.src.l3g.txt'
local tgtVocFile = 'data/vqa/train.tgt.l3g.txt'
local nMaxLength = 1
local outFile = 'data/vqa/train'

if not (th_utils.IsFile(outFile..'.src.seq.fea') and th_utils.IsFile(outFile..'.tgt.seq.fea') 
				and th_utils.IsFile(outFile..'.NegTgt.seq.fea')) then
	print('Creating Pair2Seq Feature')
	WordHash.Pair2SeqFea(inFile, srcVocFile, tgtVocFile, nMaxLength, 1, 2, outFile, 'l3g')
end

local inFile = 'data/vqa/train.src.seq.fea'
local BatchSize = batch_size
local outFile = 'data/vqa/train.src.seq.fea.t7'

if not th_utils.IsFile(outFile) then
	print('Converting Seq feature to bin: '.. inFile .. ' BatchSize: ' .. BatchSize)
	WordHash.SeqFea2Bin(inFile, BatchSize, outFile)
end

local inFile = 'data/vqa/train.tgt.seq.fea'
local BatchSize = batch_size
local outFile = 'data/vqa/train.tgt.seq.fea.t7'

if not th_utils.IsFile(outFile) then
	print('Converting Seq feature to bin: '.. inFile.. ' BatchSize: ' .. BatchSize)	
	WordHash.SeqFea2Bin(inFile, BatchSize, outFile)
end

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