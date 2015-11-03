local Batch = {}
Batch.__index = Batch

function Batch.init()
    local self = {}
    setmetatable(self, Batch)

    self.BatchSize = 0
    self.ElementSize = 0
    self.SegSize = 0
    self.m_nFeatureDim = 0
    self.m_rgFeaIdx = {}
    self.m_rgFeaVal = {}
    self.m_rgSampleIdx = {}
    self.m_rgSegIdx = {}

    return self
end

function Batch:Clear()
    self.BatchSize = 0
    self.ElementSize = 0
    self.SegSize = 0
    self.m_nFeatureDim = 0
    self.m_rgFeaIdx = {}
    self.m_rgFeaVal = {}
    self.m_rgSampleIdx = {}
    self.m_rgSegIdx = {}  
end

function Batch:LoadSeqSample(rgDict)

    local nMaxFeatureDimension = 0;

    local sid = (self.BatchSize == 0) and 0 or self.m_rgSampleIdx[self.BatchSize]
    table.insert(self.m_rgSampleIdx, sid + #rgDict)
    self.BatchSize = self.BatchSize + 1

    for _, seg in pairs(rgDict) do
        -- get the number of seg
        local count = 0
        for _,_ in pairs(seg) do
            count = count + 1
        end
        local wid = (self.SegSize == 0) and 0 or self.m_rgSegIdx[self.SegSize]
        table.insert(self.m_rgSegIdx, wid + count)
        self.SegSize = self.SegSize + 1

        for key, value in pairs(seg) do
            key = tonumber(key)
            value = tonumber(value)
            table.insert(self.m_rgFeaIdx, key)
            table.insert(self.m_rgFeaVal, value)
            self.ElementSize = self.ElementSize + 1
            if key >= nMaxFeatureDimension then
                nMaxFeatureDimension = key
            end
        end
    end

    return nMaxFeatureDimension
end

function Batch:WriteSeqSample()
    -- return the Torch.IntTensor which store one batch
    local dim = 1 + 1 + self.BatchSize + self.SegSize + self.ElementSize * 2
    local tensor = torch.IntTensor(dim)

    local idx = 1

    tensor[idx] = self.SegSize
    idx = idx + 1

    tensor[idx] = self.ElementSize
    idx = idx + 1

    for i = 1,self.BatchSize do
        tensor[idx] = self.m_rgSampleIdx[i]
        idx  = idx + 1
    end

    for i = 1, self.SegSize do
        tensor[idx] = self.m_rgSegIdx[i]
        idx = idx + 1
    end

    for i = 1, self.ElementSize do
        tensor[idx] = self.m_rgFeaIdx[i]
        idx = idx + 1
    end

    for i = 1, self.ElementSize do
        tensor[idx] = self.m_rgFeaVal[i]
        idx = idx + 1
    end

    return tensor

end

return Batch