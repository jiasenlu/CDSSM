-- Test the function. 

require 'torch'

local WordHash = require 'WordHash'
local ComputelogPD = require 'ComputelogPD'

local inFile = 'data/train.pair.tok.tsv'
local srcVocFile = 'data/l3g.txt'
local tgtVocFile = 'data/l3g.txt'
local nMaxLength = 20

local outFile = 'data/train'

--local tst = WordHash.Pair2SeqFea(inFile, srcVocFile, tgtVocFile, nMaxLength, 1, 2, outFile, 'l3g')

local inFile = 'data/train.src.seq.fea'
local BatchSize = 1024
local outFile = 'data/train.src.seq.fea.t7'

WordHash.SeqFea2Bin(inFile, BatchSize, outFile)

local inFile = 'data/train.tgt.seq.fea'
local BatchSize = 1024
local outFile = 'data/train.tgt.seq.fea.t7'

WordHash.SeqFea2Bin(inFile, BatchSize, outFile)

--local inFile = 'data/train.pair.tok.tsv'
--local outfile = 'data/train.logpD.s75'
--ComputelogPD.LargeScaleComputeLogPD(inFile, 2, 0.75, 1, outfile)

