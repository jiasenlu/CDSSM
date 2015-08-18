
local Calculate_Alpha = {}
Calculate_Alpha.__index = Calculate_Alpha

function Calculate_Alpha.calCosDist(input)
    local input1, input2, Q_Index, D_Index = input[1], input[2], input[3], input[4]
    local self = {}
    Q_Index = Q_Index:type('torch.LongTensor')
    D_Index = D_Index:type('torch.LongTensor')
    input1 = input1:index(1, Q_Index)
    input2 = input2:index(1, D_Index)
    if input1:dim() == 1 then
        input1 = input1:view(1,-1)
        input2 = input2:view(1,-1)
    end

    if not self.buffer then
        self.buffer = input1.new()
        self.w1 = input1.new()
        self.w22 = input1.new()
        self.w = input1.new()
        self.w32 = input1.new()
        self.ones = input1.new()
    end

   self.buffer:cmul(input1,input2)
   self.w1:sum(self.buffer,2)

   local epsilon = 1e-12
   self.buffer:cmul(input1,input1)
   self.w22:sum(self.buffer,2):add(epsilon)
   self.ones:resizeAs(self.w22):fill(1)
   self.w22:cdiv(self.ones, self.w22)
   self.w:resizeAs(self.w22):copy(self.w22)

   self.buffer:cmul(input2,input2)
   self.w32:sum(self.buffer,2):add(epsilon)
   self.w32:cdiv(self.ones, self.w32)
   self.w:cmul(self.w32)
   self.w:sqrt()

   local output = torch.cmul(self.w1,self.w)
   output = output:select(2,1)

   return output
end
return Calculate_Alpha