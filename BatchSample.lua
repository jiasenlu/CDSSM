local BatchSample = {}
BatchSample.__index = BatchSample

function BatchSample.BatchSample_Input(max_batch_size, maxSequence_perBatch, maxElements_perBatch)
-- allocate the space for each sample
    local self = {}
    setmetatable(self, SequenceInputStream)



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


function BatchSample.Load(mstream, expectedBatchSize)
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

    local smp_index = 0
    for i = 1, segsize do
        self.seg_Idx_Mem[i] = mstream[self.pointer]
        self.add_pointer()

        while self.sample_Idx_Mem[smp_index] <= i then
            smp_index = smp_index + 1
        end
        self.seg_Margin_Mem[i] = smp_index
        seg_Len_Mem[i] = 0
    end

    for i = 1, elementsize do
        self.fea_Idx_Mem[i] = mstream[self.pointer]
        self.add_pointer()
    end

    local sum = 0
    local seg_index = 0

    for i = 1, elementsize do
        self.fea_Value_Mem[i] = mstream[self.pointer]
        self.add_pointer()

        while (self.seg_Len_Mem[seg_index] <= i) then
            self.seg_Len_Mem[seg_index] = sum
            seg_index = seg_index + 1
            sum = 0
        end

        sum = sum + self.fea_Value_Mem[i]
    end
    self.seg_Len_Mem[seg_index] = sum

end



return BatchSample