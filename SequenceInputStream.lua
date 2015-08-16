
local BatchSample = require 'BatchSample'

local SequenceInputStream = {}
SequenceInputStream.__index = SequenceInputStream

function SequenceInputStream.init()
    local self = {}
    setmetatable(self, SequenceInputStream)
    
    self.total_batch_size = 0
    self.feature_size = 0
    self.maxElement_perBatch = 0
    self.maxSequence_perBatch = 0
    self.batch_num = 0
    self.batch_index = 0
    self.batch_size = 0
    self.last_incomplete_batch_size = 0
    self.mstream = nil
    self.dataFun = nil -- on the function, not inlcude the data

    self.sparse_flag = 0 -- initalized the sparse data or dense data.

    return self
end

function SequenceInputStream:get_dimension(data_dir, fileName, opt)

    self.batch_size = opt.batch_size
    self.mstream = torch.load(path.join(data_dir, fileName))
    self.feature_size = self.mstream[-5]
    self.total_batch_size = self.mstream[-4]
    self.maxSequence_perBatch = self.mstream[-3]
    self.maxElement_perBatch = self.mstream[-2]

    local batch_size = self.mstream[-1]

    assert(batch_size == opt.batch_size, "batch_size does not match bettwen configuration and input data!")

    self.dataFun = BatchSample.init()
    local Data = BatchSample:init_Data(batch_size, self.maxSequence_perBatch, self.maxElement_perBatch)
    self.batch_num = math.ceil(self.total_batch_size/batch_size)
    self.last_incomplete_batch_size = self.total_batch_size % batch_size
    self.batch_index = 1

    return Data
end

function SequenceInputStream:init_batch()
    self.batch_index = 1 -- set the first batch
end

function SequenceInputStream:Fill(Data, allowedFeatureDimension, opt)

    if self.batch_index > self.batch_num then
        return false, Data
    end

    Data = self:LoadDataBatch(Data, allowedFeatureDimension, opt)
    self.batch_index = self.batch_index+1
    return true, Data
end


function SequenceInputStream:LoadDataBatch(Data, allowedFeatureDimension, opt)
    local expectedBatchSize = self.batch_size
    if self.batch_index == self.batch_num and self.last_incomplete_batch_size ~= 0 then
        expectedBatchSize = self.last_incomplete_batch_size
    end
    if self.feature_size <= allowedFeatureDimension then

        Data = self.dataFun:Load(Data, self.mstream, expectedBatchSize, self.batch_index)
    end

    if opt.data_format == 0 then
    -- if the input is dense.
        Data = self.dataFun:Sparse_to_Dense(Data, expectedBatchSize, self.feature_size, opt)
    end
    return Data
end

return SequenceInputStream