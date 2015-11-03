-- useful functions enhance the lua or torch
local th_utils = {}
th_utils.__index = th_utils  

function th_utils.split(str, pat)
    
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
     table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

function th_utils.trim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function th_utils.spairs(t, order)
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    if order then
        table.sort(keys, function(a,b) return order(t,a,b) end)
    else
        table.sort(keys)
    end
    local sorted = {}
    for i = 1,#keys do
        sorted[i] = keys[i]
    end
    return sorted
end

function th_utils.IsFile(path)
  if io.open(path,'r')~=nil then
    return true
  else
    return false
  end
end

return th_utils