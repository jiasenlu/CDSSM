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
cmd:option('-objective', 'MMI', 'Currently only working NCE and MMI')
cmd:option('-loss_report', 1, '')
cmd:option('-model_path', 'cdssm/model/cdssm', '')
cmd:option('-log_file', 'cdssm/model/log.txt', '')
cmd:option('-qfile', 'cdssm/train.src.seq.fea.bin', '')
cmd:option('-dfile', 'cdssm/train.tgt.seq.fea.bin', '')
cmd:option('-nce_prob_file', 'cdssm/train.logpd.s75', '')
cmd:option('-batch_size', 512, '')
cmd:option('-ntrial', 30, '')
cmd:option('-max_iter', 60, '')
cmd:option('-gamma', 25, '')
cmd:option('-learning_rate', 0.05, '')
cmd:option('-weight_decay',0,'')
cmd:option('-momentum',0.9,'')
cmd:option('-lr', 0.005, '')
cmd:option('-lr_batch', 20, '')
cmd:option('-optim', 'sgd')

cmd:option('-loadmodel', '', '')
cmd:option('-feature_dimension_query', 0, '')
cmd:option('-feature_dimension_doc', 0, '')
cmd:option('-data_format', 0, '0=dense, 1=sparse')
cmd:option('-word_len', 20, '')
cmd:option('-mode', 'gpu', '1:gpu, 0:cpu')
cmd:option('-gpu_device', 2, '')

cmd:option('-convo', 0)
cmd:text()

opt = cmd:parse(arg)
local pridiction_only = 0

if opt.objective == 'NCE' then

end

-- the Data is the root Cdata place for storing the data.
if opt.mode == 'gpu' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
   cutorch.setDevice(opt.gpu_device)
end

local data_dir = 'data'
if pridiction_only == 0 then
local qFileNameTrain = 'vqa/train.src.seq.fea.t7'
local dFileNameTrain = 'vqa/train.tgt.seq.fea.t7'

local dssm_train = DSSM_Train.init()
local qDataTrain, dDataTrain = dssm_train:LoadTrainData(data_dir, qFileNameTrain, dFileNameTrain, nceProbDisFile, opt)

local qFileNameVal = 'vqa/val.src.seq.fea.t7'
local dFileNameVal = 'vqa/val.tgt.seq.fea.t7'
local dssm_val = DSSM_Train.init()
local qDataVal, dDataVal = dssm_val:LoadTrainData(data_dir, qFileNameVal, dFileNameVal, nceProbDisFile, opt)

dssm_train:ModelInit_FromConfig(opt)
local Min_err = 100000
local err = nil
for i = 1,opt.max_iter do
    print('iter ' .. i .. '...')
    model = dssm_train:Training(qDataTrain, dDataTrain, opt, i)
    dssm_train:reset_pointer()
    err = dssm_val:evaluate(qDataVal, dDataVal, opt, model)
    dssm_val:reset_pointer()

    if i > 20 and err < Min_err then
      Min_err = err
      torch.save('model.net', model)
    end
end
end
-- doing Prediction
local qFileNameTest = 'vqa/test.src.seq.fea.t7'
local dFileNameTest = 'vqa/test.tgt.seq.fea.t7'
local qFileNameTestId = 'data/vqa/VQA_pair_id_test.txt'
local qFileNameTestMC = 'data/vqa/VQA_pair_test.txt'
local dssm_test = DSSM_Train.init()
local qDataTest, dDataTest = dssm_test:LoadTrainData(data_dir, qFileNameTest, dFileNameTest, nceProbDisFile, opt)
print('Doing Prediction ...')
model = torch.load('model.net')
local question_id = {}
local f = assert(io.open(qFileNameTestId, "r"))

for line in f:lines() do    
  local sent = {}
  for value in line:gmatch("[^\t]+") do             
      table.insert(sent, value)
  end
  table.insert(question_id, tonumber(sent[1])) 
end 
f:close()

local MC_ans = {}
local f = assert(io.open(qFileNameTestMC, "r"))

for line in f:lines() do    
  local sent = {}
  for value in line:gmatch("[^\t]+") do             
      table.insert(sent, value)
  end

  local tmp = {}
  for i = 1, 18 do
    table.insert(tmp, sent[i+1])
  end
  table.insert(MC_ans, tmp) 
end 
f:close()

result_table = dssm_test:predict(qDataTest, dDataTest, opt, model, question_id, MC_ans)
local cjson = require 'cjson'
local encode_result = cjson.encode(result_table)

local file =  torch.DiskFile('result/result.json','w')
file:writeString(encode_result)
file:close()
