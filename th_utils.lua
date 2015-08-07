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

return th_utils