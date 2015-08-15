-- Parameter setting

require 'torch'
require 'nn'
local DSSM_Train = require 'DSSM_Train'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train CDSSM model')

cmd:text()
cmd:option('-cublas', 1, '')
cmd:option('-objective', 'NCE', '')
cmd:option('-loss_report', 1, '')
cmd:option('-model_path', 'cdssm/model/cdssm', '')
cmd:option('-log_file', 'cdssm/model/log.txt', '')
cmd:option('-qfile', 'cdssm/train.src.seq.fea.bin', '')
cmd:option('-dfile', 'cdssm/train.tgt.seq.fea.bin', '')
cmd:option('-nce_prob_file', 'cdssm/train.logpd.s75', '')
cmd:option('-batch_size', 1024, '')
cmd:option('-ntrial', 50, '')
cmd:option('-max_iter', 500, '')
cmd:option('-parm_gamma', 25, '')
cmd:option('-train_test_rate', 1, '')
cmd:option('-learning_rate', 0.02, '')

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

cmd:option('-mirror_init', 0, '')
cmd:option('-device', 'gpu', '')
cmd:option('-reject_rate', 1, '')
cmd:option('-down_rate', 1, '')
cmd:option('-accept_range', 1, '')
cmd:text()

opt = cmd:parse(arg)

if opt.objective == 'NCE' then


end

local data_dir = 'data'
local qFileName = 'train.src.seq.fea.t7'
local dFileName = 'train.tgt.seq.fea.t7'
local nceProbDisFile = 'train.logpD.s75'

local dssm_train = DSSM_Train.init()
dssm_train:LoadTrainData(data_dir, qFileName, dFileName, nceProbDisFile, opt)
dssm_train:ModelInit_FromConfig(opt)
dssm_train:Training()

--print(self.PairStream.qStream.Data.fea_Idx_Mem)

--print(dssm_train)