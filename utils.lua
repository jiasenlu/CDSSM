local utils = {}
utils.__index = utils  

local th_utils = require 'th_utils'

function utils.String2FeatStrSeq(s, N, nMaxLength, featType)
    -- convert input string to letter-n-gram sequence, each word is a letter-n-gram vector.

    -- currently only the l3g featType
    local rgw = {}
    for token in s:gmatch("[^%s]+") do
        table.insert(rgw, token)
    end

    local rgWfs = {}
    for i = 1, math.min(#rgw, nMaxLength-1) do
        if featType == 'l3g' then
            table.insert(rgWfs, utils.String2L3g(rgw[i], N))
        elseif featType == 'word' then
            -- to be implement
        end
    end

    local dict = {}

    for i = nMaxLength, #rgw do
        local tmp_dict = {}
        if featType == 'l3g' then
            tmp_dict = utils.String2L3g(rgw[i], N)
        elseif featType == 'word' then
            -- to be implement
        end

        for k, v in pairs(tmp_dict) do
            if not dict[k] then
                dict[k] = v
            else
                dict[k] = dict[k] + v
            end
        end
    end

    --print(table.getn(dict))
    if next(dict) ~= nil then
        table.insert(rgWfs, dict)
    end
    return rgWfs

end


function utils.String2L3g(s, N)
    -- convert input string to letter-n-gram vector
    -- s : input string
    -- N : ngram
    N = N - 1 -- Lua start from 1
    local wfs = {}
    for w in s:gmatch("[^%s]+") do
        local src = '#' .. w .. '#'
        for i = 1,#src-N do
            local l3g = src:sub(i, i+N)
            if not wfs[l3g] then 
                wfs[l3g] = 1
            else
                wfs[l3g] = wfs[l3g] + 1
            end
        end
    end

    return wfs
end

function utils.StrFreq2IdFreq(strFeqList, Vocab)
    ans = {}
    for _, inpDict in pairs(strFeqList) do
        dict = {}
        for k, v in pairs(inpDict) do
            if Vocab[k] then
                dict[Vocab[k]] = v
            end
        end
        table.insert(ans, dict)
    end
    return ans
end

function utils.Vector2String(vec)
    local str = ''
    local bFirst = true
    for k,v in pairs(vec) do
        if bFirst then
            str = k .. ':' .. v
            bFirst = false
        else
            str = str .. ' ' .. k .. ':' .. v
        end
    end
    return str
end

function utils.String2Vector(s)
    local vec = {}
    local rgs = th_utils.split(s, ' ')

    for _, v in pairs(rgs) do
        rgw = th_utils.split(v, ':')
        if not vec[rgw[1]] then
            vec[rgw[1]] = rgw[2]
        else
            vec[rgw[1]] = vec[rgw[1]] + rgw[2]
        end
    end 
    return vec
end

function utils.Matrix2String(mt)
    local bFirst = true
    local s = ''
    for _,vec in pairs(mt) do
        if bFirst then
            s = utils.Vector2String(vec)
            bFirst = false
        else
            s = s .. '#' .. utils.Vector2String(vec)
        end
    end
    return s
end

function utils.String2Matrix(s)
    local mt = {}
    local rgv = th_utils.split(s, '#')
    for _,vec in pairs(rgv) do
        vec = utils.String2Vector(vec)
        table.insert(mt, vec)
    end
    return mt
end


return utils 