local Normalizer = {}
Normalizer.__index = Sample

function Normalizer.CreateFeatureNormalize(type, featureSize)
    if type == 'None' then

    elseif type == 'min_max' then


    end
end




return Normalizer