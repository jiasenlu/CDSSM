local ComputelogPD = {}
ComputelogPD.__index = ComputelogPD

local th_utils = require 'th_utils'
function ComputelogPD.LargeScaleComputeLogPD(inTsv, DColumn, scale, binNum, outTsv)
    -- binNum = 1

    local colidx = DColumn
    local ht_list = {}
    for i = 1, binNum do
        -- add sub dictionary of ht_list.
        ht_list[i] = {}
    end

    local icnt = 0
    local ucnt = 0

    f = assert(io.open(inTsv, "r"))
    for line in f:lines() do 
        icnt = icnt + 1

        -- split the src and tgt by '\t'
        cols = {}
        for value in line:gmatch("[^\t]+") do 
            table.insert(cols, value)
        end

        local key = th_utils.trim(cols[colidx])

        local keyIdx = 1 -- MD5Hash key % binNum here we set 0 

        if not ht_list[keyIdx][key] then

            ht_list[keyIdx][key] = 1
            ucnt = ucnt + 1
        else
            ht_list[keyIdx][key] = ht_list[keyIdx][key] + 1
        end
    end
    f:close()
    print('total ' .. icnt .. ' lines, unique ' .. ucnt .. ' lines')

    local denom = icnt
    if scale > -0.0001 then
        print('re-scale by ' .. scale)
        denom = 0
        for _, ht in pairs(ht_list) do
            for key, value in pairs(ht)
                local x = math.pow(value, scale)
                denom = denom + x
                ht_list[key] = x
            end
        end
    end

    print(ht_list)
end



return ComputelogPD