-- Parameter setting
-- meet the 2GB out of memory again, it seems that the nested class structure 
-- didn't fit luaJit well. Split the data from the function. 

require 'torch'
require 'nn'
local tds = require 'tds'
local DSSM_Train = require 'DSSM_Train'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train CDSSM model')

cmd:text()
cmd:option('-cublas', 1, '')
cmd:option('-objective', 'MMI', 'Currently only working NCE and MMI')
cmd:option('-loss_report', 1, '')
cmd:option('-model_path', 'cdssm/model/cdssm', '')
cmd:option('-log_file', 'cdssm/model/log.txt', '')
cmd:option('-qfile', 'cdssm/train.src.seq.fea.bin', '')
cmd:option('-dfile', 'cdssm/train.tgt.seq.fea.bin', '')
cmd:option('-nce_prob_file', 'cdssm/train.logpd.s75', '')
cmd:option('-batch_size', 1024, '')
cmd:option('-ntrial', 50, '')
cmd:option('-neg_static_sample', 0, '')
cmd:option('-max_iter', 500, '')
cmd:option('-gamma', 25, '')
cmd:option('-train_test_rate', 1, '')
cmd:option('-learning_rate', 1e-1, '')
cmd:option('-weight_decay',1e-2,'')
cmd:option('-momentum',0,'')

cmd:option('-source_layer_dim', '1000 - 300', '')
cmd:option('-source_layer_sigma', '0.1 - 0.1', '')
cmd:option('-source_activation', '1 - 1', '0:Linear, 1:Tanh, 2: rectified')
cmd:option('-source_arch', '1 - 0', '0: Fully Connected, 1: Convolutional')
cmd:option('-source_arch_wind', '3 - 1', '')

cmd:option('-target_layer_dim', '1000 - 300', '')
cmd:option('-target_layer_sigma', '0.1 - 0.1', '')
cmd:option('-target_activation', '1 - 1', '0:Linear, 1:Tanh, 2: rectified')
cmd:option('-target_arch', '1 - 0', '0: Fully Connected, 1: Convolutional')
cmd:option('-target_arch_wind', '3 - 1', '')


cmd:option('-feature_dimension_query', 0, '')

cmd:option('-feature_dimension_doc', 0, '')

cmd:option('-data_format', 0, '0=dense, 1=sparse')
cmd:option('-word_len', 20, '')
cmd:option('-mirror_init', 0, '')
cmd:option('-device', 'gpu', '')
cmd:option('-reject_rate', 1, '')
cmd:option('-down_rate', 1, '')
cmd:option('-accept_range', 1, '')
cmd:option('-mode', 0, '1:gpu, 0:cpu')
cmd:text()

opt = cmd:parse(arg)

if opt.objective == 'NCE' then

end

local data_dir = 'data'
local qFileName = 'train.src.seq.fea.t7'
local dFileName = 'train.tgt.seq.fea.t7'
local nceProbDisFile = 'train.logpD.s75'

-- the Data is the root Cdata place for storing the data.
if opt.mode == 1 then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
   cutorch.setDevice(opt.gpu_device)
end

local dssm_train = DSSM_Train.init()
local qData, dData = dssm_train:LoadTrainData(data_dir, qFileName, dFileName, nceProbDisFile, opt)
dssm_train:ModelInit_FromConfig(opt)
for i = 1,20 do
    dssm_train:Training(qData, dData, opt)
    dssm_train:reset_pointer()
end
--print(self.PairStream.qStream.Data.fea_Idx_Mem)

--print(dssm_train)