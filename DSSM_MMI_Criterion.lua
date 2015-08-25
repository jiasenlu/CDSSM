local DSSM_MMI_Criterion, parent = torch.class('nn.DSSM_MMI_Criterion', 'nn.Criterion')

local Calculate_Alpha = require 'Calculate_Alpha'

function DSSM_MMI_Criterion:__init(batch_size, nTrail, gamma)
    parent.__init(self)
    self.batch_size = batch_size
    self.nTrail = nTrail
    self.gamma = gamma
    -- do negative sampling
    self.D_negtive_array  = torch.IntTensor(batch_size * nTrail)

    for i = 1, nTrail do
        local randpos = torch.random(0.8*batch_size) + math.floor(0.1 * batch_size)
        for k = 1, batch_size do
            local bs = (randpos + k) % batch_size + 1
            self.D_negtive_array [(i-1)*batch_size + k] = bs
        end
    end

    -- concatinate with the postive.
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function DSSM_MMI_Criterion:updateOutput(input)
    
    local Q_input, D_input = input[1],input[2]
    self.dimension = Q_input:size()[2]

    self.input1 = Q_input:repeatTensor(self.nTrail+1,1)

    self.input2 = torch.Tensor((self.nTrail+1)*self.batch_size, self.dimension):zero()

    self.input2:sub(1, self.batch_size):copy(D_input) -- copy the positive

    for i = 1, self.nTrail * self.batch_size do
        self.input2:sub(i+self.batch_size, i+self.batch_size):copy(D_input:sub(self.D_negtive_array[i],self.D_negtive_array[i]))
    end
    

    -- 1: calculate the cosine distance for positive and negtive array

    if not self.buffer then
        self.buffer = self.input1.new()
        self.w1 = self.input1.new()
        self.w22 = self.input1.new()
        self.w = self.input1.new()
        self.w32 = self.input1.new()
        self.ones = self.input1.new()
    end

    self.buffer:cmul(self.input1, self.input2)
    self.w1:sum(self.buffer,2)

    local epsilon = 1e-12
    self.buffer:cmul(self.input1,self.input1)
    self.w22:sum(self.buffer,2):add(epsilon)
    self.ones:resizeAs(self.w22):fill(1)
    self.w22:cdiv(self.ones, self.w22)
    self.w:resizeAs(self.w22):copy(self.w22)

    self.buffer:cmul(self.input2,self.input2)
    self.w32:sum(self.buffer,2):add(epsilon)
    self.w32:cdiv(self.ones, self.w32)
    self.w:cmul(self.w32)
    self.w:sqrt()

    self.alpha_buffer= torch.cmul(self.w1,self.w)
    self.alpha_buffer = self.alpha_buffer:select(2,1)

    -- 2: calculate the alpha 
    self.alpha_buffer = Calculate_Alpha.cal_alpha(self.alpha_buffer, self.nTrail, self.batch_size, self.gamma)
    self.alpha_buffer = Calculate_Alpha.cal_alpha_sum(self.alpha_buffer, self.nTrail, self.batch_size, self.gamma,1)
    self.alpha_buffer = Calculate_Alpha.cal_alpha_norm(self.alpha_buffer, self.nTrail, self.batch_size, self.gamma)
    self.alpha_buffer = Calculate_Alpha.cal_alpha_sum(self.alpha_buffer, self.nTrail, self.batch_size, self.gamma,0)

    -- 3: calculate the loss
    local err = 0
    local eps = 1.4e-45
    for i = 1, self.batch_size do
        err = err + math.log(math.max(eps, (1+self.alpha_buffer[i] / math.max(self.gamma - self.alpha_buffer[i], eps))))
    end
    return err 
end

function DSSM_MMI_Criterion:updateGradInput()
    -- here, we didn't use the fomula in the paper, we jsut
    -- j is the number of negative sampling
    -- gradInput = Sum_j[alpha_buffer[j] + (delta_postivePair - delta_negtivePair)]
   local gw1 = torch.Tensor()
   local gw2 = torch.Tensor()
   gw1:resizeAs(self.input1):copy(self.input2)
   gw2:resizeAs(self.input1):copy(self.input1)
   self.w = self.w:expandAs(self.input1)
   self.buffer:cmul(self.w1,self.w22)
   self.buffer = self.buffer:expandAs(self.input1)
   gw1:addcmul(-1,self.buffer,self.input1)
   gw1:cmul(self.w)

   local temp = gw1:sub(1, self.batch_size)

   gw1 = torch.add(temp:repeatTensor(self.nTrail+1,1), -gw1)

   self.buffer:cmul(self.w1,self.w32)
   self.buffer = self.buffer:expandAs(self.input1)
   gw2:addcmul(-1,self.buffer,self.input2)
   gw2:cmul(self.w)

   temp = gw2:sub(1, self.batch_size)
   
   gw2 = torch.add(temp:repeatTensor(self.nTrail+1,1), -gw2)

    self.gradInput = {torch.Tensor, torch.Tensor}
    self.gradInput[1] = torch.Tensor(self.dimension):zero()
    self.gradInput[2] = torch.Tensor(self.dimension):zero()

   for i = 1, self.batch_size do
        for j = 1, self.nTrail do
            self.gradInput[1]:add(torch.mul(gw1[j*self.batch_size+i], self.alpha_buffer[j*self.batch_size+i] / self.batch_size ))
            self.gradInput[2]:add(torch.mul(gw2[j*self.batch_size+i], self.alpha_buffer[j*self.batch_size+i] / self.batch_size ))
        end
   end

   return self.gradInput

end
