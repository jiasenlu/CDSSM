
local PairInputStream = require 'PairInputStream'
local Normalizer = require 'Normalizer'
local tds = require 'tds'
require 'nngraph'
require 'nn'
require 'DSSM_MMI_Criterion'
require 'optim'
local th_utils = require 'th_utils'
local DSSM_Predict = require 'DSSM_Predict'
local DSSM_Train = {}
DSSM_Train.__index = DSSM_Train

function DSSM_Train.init()
    self = {}
    setmetatable(self, DSSM_Train)

    self.PairStream = nil
    self.PairValidStream = nil
    self.SrcNorm = nil
    self.TgtNorm = nil
    self.pairTrainFileIdx = 0

    self.model = nil
    return self
end

function DSSM_Train:reset_pointer()
    self.PairStream.qStream.dataFun.pointer = 1
    self.PairStream.dStream.dataFun.pointer = 1
end
function DSSM_Train:LoadTrainData(data_dir, qFileName, dFileName, nceProbDisFile, opt)
    -- doing shuffle and ...

    local qData, dData = self:LoadPairDataAtIdx(data_dir, qFileName, dFileName, nceProbDisFile, opt)

    print('Loading training doc query stream done')
    return qData, dData
end
function DSSM_Train:LoadPairDataAtIdx(data_dir, qFileName, dFileName, nceProbDisFile, opt)

    self.PairStream = PairInputStream.init()

    local qData, dData = self.PairStream:Load_Train_PairData(data_dir, qFileName, dFileName, nceProbDisFile, opt)

    self.ScrNorm = Normalizer.CreateFeatureNormalize(opt.Q_FeaNorm, self.PairStream.qStream.feature_size)
    self.TgtNorm = Normalizer.CreateFeatureNormalize(opt.D_FeaNorm, self.PairStream.dStream.feature_size)

    -- PairStream:initFeatureNorm()
    self.pairTrainFileIdx = 0

    return qData, dData
end

function DSSM_Train:ModelInit_FromConfig(opt)

    -- we first try to build fix structure model
    print('Initializing the Model...')
    local Q_feature_size = self.PairStream.qStream.feature_size
    local D_feature_size = self.PairStream.dStream.feature_size
    print(Q_feature_size)
    print(D_feature_size)

    local wind_size = 3

    local model
    if opt.convo == 0 then
        local Q_layer = nn.Sequential()
        Q_layer:add(nn.Linear(Q_feature_size, 1000))
        Q_layer:add(nn.Tanh())
        Q_layer:add(nn.Dropout(0.5))
        Q_layer:add(nn.Linear(1000, 300))
        Q_layer:add(nn.Tanh())


        local D_layer = nn.Sequential()
        D_layer:add(nn.Linear(D_feature_size, 1000))
        D_layer:add(nn.Tanh())
        D_layer:add(nn.Dropout(0.5))        
        D_layer:add(nn.Linear(1000, 300))
        D_layer:add(nn.Tanh())

        model = nn.ParallelTable()
        model:add(Q_layer)
        model:add(D_layer)
    else

        local Q_layer = nn.Sequential()
        Q_layer:add(nn.TemporalConvolution(Q_feature_size * wind_size, 1000, 1))
        Q_layer:add(nn.Tanh())
        Q_layer:add(nn.TemporalMaxPooling(18))
        Q_layer:add(nn.Reshape(1000, true))
        Q_layer:add(nn.Linear(1000, 300))
        
        local D_layer = nn.Sequential()
        D_layer:add(nn.TemporalConvolution(D_feature_size * wind_size, 1000, 1))
        D_layer:add(nn.Tanh())
        D_layer:add(nn.TemporalMaxPooling(18))
        D_layer:add(nn.Reshape(1000, true))
        D_layer:add(nn.Linear(1000, 300))
        
        model = nn.ParallelTable()
        model:add(Q_layer)
        model:add(D_layer)
    end

    self.model = model
    if opt.mode == 'gpu' then
        self.model:cuda()
    end
--     
    parameters,gradParameters = self.model:getParameters()
    --print(Q_feature_size)
    --print(parameters:size())
    if opt.optim == 'sgd' then
        self.optimState = {
          learningRate = opt.learning_rate,
          weightDecay = opt.weight_decay,
          momentum = opt.momentum,
          learningRateDecay = 1e-7
        }
        self.optimMethod = optim.sgd
    end
    if opt.optim == 'rmsprop' then
        self.optimState = {
          learningRate = nil,
          alpha = nil,
          epsilon = nil
        }
        self.optimMethod = optim.rmsprop        
    end

end


function DSSM_Train:Training(qData, dData, opt, epoch)
    
    if epoch > opt.lr_batch then
        self.optimState.learningRate = opt.lr
    end
    print('learning rate: '..self.optimState.learningRate)
    if opt.loadmodel ~= '' then
        self.model = torch.load(opt.loadmodel)
    end
    self.model:training()
    self.PairStream:Init_Batch()
    local trainingLoss = 0
    local total_err = 0

    for i = 1, self.PairStream.qStream.batch_num do
        local flag, qData, dData = self.PairStream:Next_Batch(qData, dData, srcNorm, tgtNorm, opt)

        xlua.progress(i, self.PairStream.qStream.batch_num)
        -- doing the forward process
        local qTensor, dTensor
        if opt.mode == 'gpu' then
            qTensor = qData.data_matrix:cuda()
            dTensor = dData.data_matrix:cuda()   
        else
            qTensor = qData.data_matrix:double()
            dTensor = dData.data_matrix:double()     
        end
        -- negtive samping index
        local batch_size = self.PairStream.qStream.dataFun.batch_size
        local feval = function(x)
                if x ~= parameters then
                    parameters:copy(x)
                end
                gradParameters:zero()
                -- get the cosine similarity for all positive and negtive.
                local output = self.model:forward({qTensor, dTensor})
                -- initialize the criterion function here, in order we can intilized the negative sampling 
                -- differently for each batch

                if opt.objective == 'NCE' then
                -- load the doc distance for NCE Training.
                end
                --if opt.mode == 'gpu' then
                --    output = {output[1]:double(), output[2]:double()}
                --end
                if opt.objective == 'MMI' then
                    self.criterion = nn.DSSM_MMI_Criterion(batch_size, opt.ntrial, opt.gamma, opt.batch_size)
                end
                if opt.mode == 'gpu' then
                    self.criterion:cuda()
                end

                --print(output)
                -- forward the criterion
                local err = self.criterion:updateOutput(output)
                local do_df = self.criterion:updateGradInput()
                self.model:backward({qTensor, dTensor}, do_df)
                gradParameters:div(batch_size)
                total_err = total_err + err
                err = err / batch_size
                return err, gradParameters
            end
            self.optimMethod(feval, parameters, self.optimState)
    end
    print('training Loss: ' .. total_err / self.PairStream.qStream.batch_num)
    

    return self.model
end


function DSSM_Train:evaluate(qData, dData, opt, model)
    model:evaluate()

    -- load label
    self.PairStream:Init_Batch()
    local trainingLoss = 0
    local total_err = 0
    -- get the encoding of each answer.    
    
    for i = 1, self.PairStream.qStream.batch_num do
        xlua.progress(i, self.PairStream.qStream.batch_num)
        local flag, qData, dData = self.PairStream:Next_Batch(qData, dData, srcNorm, tgtNorm, opt)

        -- doing the forward process
        local qTensor, dTensor
        if opt.mode == 'gpu' then
            qTensor = qData.data_matrix:cuda()
            dTensor = dData.data_matrix:cuda()
        else
            qTensor = qData.data_matrix:double()
            dTensor = dData.data_matrix:double()     
        end
        -- negtive samping index
        local batch_size = self.PairStream.qStream.dataFun.batch_size
        -- get the cosine similarity for all positive and negtive.
        -- 
        local output = model:forward({qTensor, dTensor})

        if opt.objective == 'NCE' then
        -- load the doc distance for NCE Training.
        end
        if opt.objective == 'MMI' then
            self.criterion = nn.DSSM_MMI_Criterion(batch_size, opt.ntrial, opt.gamma, opt.batch_size)
        end

        -- forward the criterion
        local err = self.criterion:updateOutput(output)
        total_err = total_err + err        
        err = err / batch_size
    end
    total_err = total_err / self.PairStream.qStream.batch_num
    print('validation loss: ' .. total_err)
    return total_err
end


function DSSM_Train:predict(qData, dData, opt, model, question_id, MC_ans)
    self.model = model
    self.model:evaluate()

    -- load label
    self.PairStream:Init_Batch()
    -- get the encoding of each answer.    
    
    local result_table = {}
    count = 1
    for i = 1,self.PairStream.qStream.batch_num do
        xlua.progress(i, self.PairStream.qStream.batch_num)
        local flag, qData, dData = self.PairStream:Next_Batch(qData, dData, srcNorm, tgtNorm, opt)

        -- doing the forward process
        local qTensor, dTensor
        if opt.mode == 'gpu' then
            qTensor = qData.data_matrix:cuda()
            dTensor = dData.data_matrix:cuda()
        else
            qTensor = qData.data_matrix:double()
            dTensor = dData.data_matrix:double()     
        end
        -- negtive samping index
        local batch_size = self.PairStream.qStream.dataFun.batch_size
        -- get the cosine similarity for all positive and negtive.
        -- 
        local output = self.model:forward({qTensor, dTensor})

        if opt.objective == 'NCE' then
        -- load the doc distance for NCE Training.
        end
        prediction = DSSM_Predict:updateOutput(output, 17)

        for j = 1, batch_size do
            local tmp = {}
            tmp['question_id'] = question_id[count]
            tmp['answer'] = MC_ans[count][prediction[j][1]]
            table.insert(result_table, tmp)
            count = count + 1            
        end
    end
    return result_table
end
return DSSM_Train
