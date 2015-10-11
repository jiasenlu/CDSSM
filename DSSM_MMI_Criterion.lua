local DSSM_MMI_Criterion, parent = torch.class('nn.DSSM_MMI_Criterion', 'nn.Criterion')

local Calculate_Alpha = require 'Calculate_Alpha'

function DSSM_MMI_Criterion:__init(batch_size, nTrail, gamma, BATCH_SIZE)
    parent.__init(self)
    self.batch_size = batch_size
    self.nTrail = nTrail
    self.gamma = gamma
    -- do negative sampling
    self.BATCH_SIZE = BATCH_SIZE
    self.D_negtive_array  = torch.IntTensor(batch_size * nTrail)

    for i = 1, nTrail do
        local randpos = torch.random(0.8*batch_size) + math.floor(0.1 * batch_size)
        for k = 1, batch_size do
            local bs = (randpos + k) % batch_size + 1
            self.D_negtive_array[(i-1)*batch_size + k] = bs
        end
    end
    
    --self.D_negtive_array = torch.load('negSampline')


    -- inverse negtive array
    self.D_inver_negtive_index_array = torch.IntTensor(batch_size * nTrail):zero()
    self.D_inver_negtive_value_array = torch.IntTensor(batch_size * nTrail):zero()

    for i = 1, nTrail do
        local mlist = torch.IntTensor(batch_size):zero()

        for k = 1, batch_size do
            local bs = self.D_negtive_array[(i-1) * batch_size + k]
            mlist[bs] = k
        end

        local ptotal = 0
        local pindex = 1
        for k = 1, batch_size do
            self.D_inver_negtive_value_array[(i-1)*batch_size+pindex] = mlist[k]
            pindex = pindex + 1
            self.D_inver_negtive_index_array[(i-1)*batch_size+k] = ptotal + 1
            ptotal = self.D_inver_negtive_index_array[(i-1)*batch_size+k]

        end
    end

    -- concatinate with the postive.
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function DSSM_MMI_Criterion:updateOutput(input)
    
    local Q_input, D_input = input[1],input[2]
    --Q_input:tanh()
    --D_input:tanh()

    self.dimension = Q_input:size()[2]
    self.input1 = Q_input:repeatTensor(self.nTrail+1,1)
    self.input2 = torch.Tensor((self.nTrail+1)*self.batch_size, self.dimension):zero():typeAs(D_input)
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
    -- self.w = b*c
    -- self.w32 = c^2
    -- self.w22 = b^2
    -- self.w1 = a
    self.alpha_buffer= torch.cmul(self.w1,self.w)
    self.alpha_buffer = self.alpha_buffer:select(2,1)

    -- 2: calculate the alpha 
    self.alpha_buffer = Calculate_Alpha.cal_alpha(self.alpha_buffer, self.nTrail, self.batch_size, self.gamma) -- check right.
    self.alpha_buffer = Calculate_Alpha.cal_alpha_sum(self.alpha_buffer, self.nTrail, self.batch_size, self.gamma,1)
    self.alpha_buffer = Calculate_Alpha.cal_alpha_norm(self.alpha_buffer, self.nTrail, self.batch_size, self.gamma)
    self.alpha_buffer = Calculate_Alpha.cal_alpha_sum(self.alpha_buffer, self.nTrail, self.batch_size, self.gamma,0)
    
    -- 3: calculate the loss
    local err = 0
    local eps = 1.4e-45
    for i = 1, self.batch_size do
        err = err + math.log(math.max(eps, (1+self.alpha_buffer[i] / math.max(self.gamma - self.alpha_buffer[i], eps))))
        --err = err + math.log(math.max(self.alpha_buffer[i])
    end
    return err 
end

function DSSM_MMI_Criterion:updateGradInput()

   local gw1 = torch.Tensor():typeAs(self.input1)
   local gw2 = torch.Tensor():typeAs(self.input2)
   gw1:resizeAs(self.input1):zero()
   gw2:resizeAs(self.input2):zero()

   self.w = self.w:expandAs(self.input1)
   self.w22 = self.w22:expandAs(self.input1)
   self.w32 = self.w32:expandAs(self.input1)
   self.w1 = self.w1:expandAs(self.input1)

   gw1:cmul(self.input2, self.w) -- bc * Yd

   -- q * a * (b*b*b*c)
   self.buffer:cmul(self.w, self.w22)  -- (b*b*b*c)
   self.buffer:cmul(self.w1, self.buffer) -- a * ...
   self.buffer:cmul(self.input1, self.buffer) -- Yq * ...

   gw1:add(-self.buffer) -- bc * Yd - a * (b*b*b*c) * Yq

   -- for deriv_d
   -- q / (b * c)

   gw2:cmul(self.input1, self.w)
   -- d * a / (b*c*c*c)
   self.buffer:cmul(self.w, self.w32)
   self.buffer:cmul(self.w1, self.buffer)
   self.buffer:cmul(self.input2, self.buffer)

   -- q / (b*c) - d*a / (bbbc)
   gw2:add(-self.buffer)

  -- matrix_weightAdd
  -- for i = 1, batchsize*dimension
  local ngw1 = torch.Tensor(self.batch_size, self.dimension):zero():typeAs(gw1)
  local ngw2 = torch.Tensor(self.batch_size, self.dimension):zero():typeAs(gw2)

  ngw1:cmul(gw1:sub(1, self.batch_size), self.alpha_buffer:sub(1, self.batch_size):view(-1,1):expandAs(ngw1))

  for i = 1, self.nTrail do
    ngw1:addcmul(-1, gw1:sub(i*self.batch_size+1, (i+1)*self.batch_size), 
      self.alpha_buffer:sub(i*self.batch_size+1, (i+1)*self.batch_size):view(-1,1):expandAs(ngw1))  
  end


  ngw2:cmul(gw2:sub(1, self.batch_size), self.alpha_buffer:sub(1, self.batch_size):view(-1,1):expandAs(ngw2))
  for i = 1, self.batch_size do
    for j = 1, self.nTrail do
      local col = self.D_inver_negtive_index_array[(j-1)*self.batch_size + i]
      local row = self.D_inver_negtive_value_array[(j-1)*self.batch_size + col]
      ngw2:sub(i,i):addcmul(-1, gw2:sub(j*self.batch_size + row, j*self.batch_size + row),  
        self.alpha_buffer:sub(j*self.batch_size + row, j*self.batch_size + row):view(-1,1):expandAs(ngw2:sub(i,i)))
    end
  end

  self.gradInput = {-ngw1, -ngw2}

  return self.gradInput

end
