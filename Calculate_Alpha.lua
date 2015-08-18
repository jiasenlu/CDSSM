
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

function Calculate_Alpha.cal_alpha(alpha, nTrial, batchsize, gamma)
    -- alpha[negtive] = exp(-gamma * (alpha[positive] - alpha[negative])
    local positive_array = torch.range(1,batchsize):type('torch.IntTensor')  
    local pos_alpha = alpha:sub(1,batchsize)
    local neg_alpha = alpha:sub(batchsize+1, -1)
    positive_array = positive_array:repeatTensor(nTrial) -- replicate the nTrail times
    positive_array = positive_array:type('torch.LongTensor')

    local pos_alpha_replicate = pos_alpha:index(1,positive_array)

    neg_alpha = torch.add(pos_alpha_replicate, - neg_alpha) 
    neg_alpha = torch.exp(neg_alpha * gamma)
    
    local new_alpha = torch.cat(pos_alpha, neg_alpha)
    return new_alpha
end

function Calculate_Alpha.cal_alpha_sum(alpha, nTrial, batchsize, gamma, init)
    -- alpha[postive] = init + alpha[postive + negative]
    local pos_alpha = alpha:sub(1,batchsize)
    local neg_alpha = alpha:sub(batchsize+1, -1)    
    local neg_alpha_split = neg_alpha:split(batchsize, 1)
    for i = 1, nTrial do
      pos_alpha:add(neg_alpha_split[i], init)
    end

    local new_alpha = torch.cat(pos_alpha, neg_alpha)
    return new_alpha
end

function Calculate_Alpha.cal_alpha_norm(alpha, nTrial, batchsize, gamma)
    -- alpha[negative] = gamma * alpha[negative]./ alpha[positive]
    local pos_alpha = alpha:sub(1,batchsize)
    local neg_alpha = alpha:sub(batchsize+1, -1)
    local positive_array = torch.range(1,batchsize):type('torch.IntTensor')  
    positive_array = positive_array:repeatTensor(nTrial) -- replicate the nTrail times
    positive_array = positive_array:type('torch.LongTensor')
    local pos_alpha_replicate = pos_alpha:index(1,positive_array)

    neg_alpha = torch.cdiv(neg_alpha, pos_alpha_replicate)
    neg_alpha = gamma * neg_alpha
    local new_alpha = torch.cat(pos_alpha, neg_alpha)

    return new_alpha

end
return Calculate_Alpha