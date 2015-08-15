local BatchSample = {}
BatchSample.__index = BatchSample

function BatchSample.init(max_batch_size, maxSequence_perBatch, maxElements_perBatch)
-- allocate the space for each sample
    local self = {}
    setmetatable(self, BatchSample)



    self.sample_Idx_Mem = torch.IntTensor(max_batch_size)
    self.seg_Idx_Mem = torch.IntTensor(maxSequence_perBatch)
    self.seg_Margin_Mem = torch.IntTensor(maxSequence_perBatch)
    self.seg_Len_Mem = torch.IntTensor(maxSequence_perBatch)
    self.fea_Idx_Mem = torch.IntTensor(maxElements_perBatch)
    self.fea_Value_Mem = torch.FloatTensor(maxElements_perBatch)

    self.pointer = 1

    self.add_pointer = function() self.pointer = self.pointer + 1 end
    return self
end


function BatchSample:Load(mstream, expectedBatchSize)
    -- load into 1D sequence vector.
    local batch_size = expectedBatchSize
    local segsize = mstream[self.pointer]
    self.add_pointer()

    local elementsize = mstream[self.pointer]
    self.add_pointer()

    for i = 1, batch_size do
        self.sample_Idx_Mem[i] = mstream[self.pointer]
        self.add_pointer()
    end

    --print(self.sample_Idx_Mem) test right

    local smp_index = 1
    for i = 1, segsize do
        self.seg_Idx_Mem[i] = mstream[self.pointer]
        self.add_pointer()
        while self.sample_Idx_Mem[smp_index] < i do
            smp_index = smp_index + 1
        end
        self.seg_Margin_Mem[i] = smp_index
        self.seg_Len_Mem[i] = 0
    end

    -- print(self.seg_Margin_Mem) test right

    for i = 1, elementsize do
        self.fea_Idx_Mem[i] = mstream[self.pointer]
        self.add_pointer()
    end
    -- print(self.fea_Idx_Mem) the word level hash order maybe different-- mater?


    local sum = 0
    local seg_index = 1

    for i = 1, elementsize do
        self.fea_Value_Mem[i] = mstream[self.pointer]
        self.add_pointer()
        while (self.seg_Idx_Mem[seg_index] < i) do
            self.seg_Len_Mem[seg_index] = sum
            seg_index = seg_index + 1
            sum = 0
        end

        sum = sum + self.fea_Value_Mem[i]
    end
    self.seg_Len_Mem[seg_index] = sum
    -- print(self.seg_Len_Mem) checked right
end

return BatchSample