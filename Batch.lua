local Batch = {}
Batch.__index = Batch

function Batch.init()
    local self = {}
    setmetatable(self, Batch)

    self.BatchSize = function () return #self.m_rgSampleIdx end
    self.ElementSize = function () return #m_rgFeaIdx end
    self.SegSize = function () return #self.m_rgSegIdx end
    self.m_nFeatureDim = 0
    self.m_rgFeaIdx = {}
    self.m_rgFeaVal = {}
    self.m_rgSampleIdx = {}
    self.m_rgSegIdx = {}

    return self
end

function Batch:Clear()
    self.m_nFeatureDim = 0
    self.m_rgFeaIdx = {}
    self.m_rgFeaVal = {}
    self.m_rgSampleIdx = {}
    self.m_rgSegIdx = {}    
end

function Batch:LoadSeqSample(rgDict)

    local nMaxFeatureDimension = 0;

    local sid = (self.BatchSize() == 0) and 0 or self.m_rgSampleIdx[self.BatchSize()]
    table.insert(self.m_rgSampleIdx, sid)

    for _, seg in pairs(rgDict) do
        -- get the number of seg
        local count = 0
        for _,_ in pairs(seg) do
            count = count + 1
        end
        local wid = (self.SegSize() == 0) and 0 or self.m_rgSegIdx[self.SegSize()]
        table.insert(self.m_rgSegIdx, wid + count)
        for key, value in pairs(seg) do
            key = tonumber(key)
            value = tonumber(value)
            table.insert(self.m_rgFeaIdx, key)
            table.insert(self.m_rgFeaVal, value)
            if key >= nMaxFeatureDimension then
                nMaxFeatureDimension = key
            end
        end
    end

    return nMaxFeatureDimension
end

return Batch