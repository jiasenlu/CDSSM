local BatchSample = {}
BatchSample.__index = BatchSample
local tds = require 'tds'

function BatchSample.init()
-- allocate the space for each sample
    local self = {}
    setmetatable(self, BatchSample)

    self.pointer = 1

    self.add_pointer = function() self.pointer = self.pointer + 1 end
    return self, Data
end

function BatchSample:init_Data(max_batch_size, maxSequence_perBatch, maxElements_perBatch)
    local Data = tds.hash()

    Data.sample_Idx_Mem = torch.IntTensor(max_batch_size)
    Data.seg_Idx_Mem = torch.IntTensor(maxSequence_perBatch)
    Data.seg_Margin_Mem = torch.IntTensor(maxSequence_perBatch)
    Data.seg_Len_Mem = torch.IntTensor(maxSequence_perBatch)
    Data.fea_Idx_Mem = torch.IntTensor(maxElements_perBatch)
    Data.fea_Value_Mem = torch.FloatTensor(maxElements_perBatch)
    Data.data_matrix = nil
    return Data
end

function BatchSample:Load(Data, mstream, expectedBatchSize, index)
    -- load into 1D sequence vector.
    local batch_size = expectedBatchSize
    local segsize = mstream[self.pointer]
    self.add_pointer()

    local elementsize = mstream[self.pointer]
    self.add_pointer()

    --print(batch_size, segsize, elementsize)
    

    for i = 1, batch_size do
        Data.sample_Idx_Mem[i] = mstream[self.pointer]
        self.add_pointer()
    end

    --print(self.sample_Idx_Mem) --test right

    local smp_index = 1
    for i = 1, segsize do
        Data.seg_Idx_Mem[i] = mstream[self.pointer]
        self.add_pointer()
        while Data.sample_Idx_Mem[smp_index] < i do
            smp_index = smp_index + 1
        end
        Data.seg_Margin_Mem[i] = smp_index
        Data.seg_Len_Mem[i] = 0
    end

    --print(self.seg_Idx_Mem) --test right

    for i = 1, elementsize do
        Data.fea_Idx_Mem[i] = mstream[self.pointer]
        self.add_pointer()
    end
    --print(self.fea_Idx_Mem)-- the word level hash order maybe different-- mater?


    local sum = 0
    local seg_index = 1

    for i = 1, elementsize do
        Data.fea_Value_Mem[i] = mstream[self.pointer]
        self.add_pointer()
        while (Data.seg_Idx_Mem[seg_index] < i) do
            Data.seg_Len_Mem[seg_index] = sum
            seg_index = seg_index + 1
            sum = 0
        end

        sum = sum + Data.fea_Value_Mem[i]
    end
    Data.seg_Len_Mem[seg_index] = sum
    --print(self.seg_Len_Mem) --checked right
    return Data
end

function BatchSample:Sparse_to_Dense(Data, expectedBatchSize, feature_size, opt)
    -- transfer the sparse vector to dense matrix batch.
    local win_size = 3
    local seg_len = opt.word_len  -- we need to fix the seg_len here, the last dim is bow encodding.
    -- initialize the tensor.
    local data_matrix = torch.IntTensor(expectedBatchSize, seg_len, feature_size)
    local zero_matrix = torch.IntTensor(1):zero()
    Data.sample_Idx_Mem = torch.cat(zero_matrix, Data.sample_Idx_Mem)
    Data.seg_Idx_Mem = torch.cat(zero_matrix, Data.seg_Idx_Mem)

    for i = 1, expectedBatchSize do
        -- for each sentence
        local sample_Idx_start = Data.sample_Idx_Mem[i]
        local sample_Idx_end = Data.sample_Idx_Mem[i+1]
        for j = sample_Idx_start+1, sample_Idx_end do
            -- for each words
            local seg_Idx_start = Data.seg_Idx_Mem[j]
            local seg_Idx_end = Data.seg_Idx_Mem[j+1]
            for k = seg_Idx_start+1, seg_Idx_end do
                local idx = Data.fea_Idx_Mem[k]
                local value = Data.fea_Value_Mem[k]
                data_matrix[i][j-sample_Idx_start][idx] = value
            end
        end
    end    
    -- this is only when window size == 1, 
    Data.data_matrix = data_matrix:sub(1,expectedBatchSize, 1, seg_len - win_size +1)
    
    for i = 2,win_size do
        Data.data_matrix = torch.cat(Data.data_matrix, data_matrix:sub(1, expectedBatchSize, i, seg_len - win_size + i))
    end
    return Data
end


return BatchSample