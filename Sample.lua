local Sample = {}
Sample.__index = Sample

function Sample.BatchSample_Input(max_batch_size, maxSequence_perBatch, maxElements_perBatch)
-- allocate the space for each sample
    local self = {}
    setmetatable(self, SequenceInputStream)

    self.sample_idx = torch.IntTensor(max_batch_size)
    self.seg_idx = torch.IntTensor(maxSequence_perBatch)
    self.seg_margin = torch.IntTensor(maxSequence_perBatch)
    self.seg_len = torch.IntTensor(maxSequence_perBatch)
    self.fea_idx = torch.IntTensor(maxElements_perBatch)
    self.fea_value = torch.FloatTensor(maxElements_perBatch)

    return self
end

return Sample