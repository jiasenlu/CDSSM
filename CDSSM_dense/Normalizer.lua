local Normalizer = {}
Normalizer.__index = Sample
function Normalizer.init(featureDim)
    self = {}
    setmetatable(self, Normalizer)

    self.featureDim = 0

    return self

end

function Normalizer.CreateFeatureNormalize(type, featureSize)
    local norm = nil
    if type == 'None' then
        norm = Normalizer.init(featureSize)

    elseif type == 'min_max' then

    end

    return norm
end

return Normalizer