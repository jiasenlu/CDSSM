local DSSM_MMI_Criterion, parent = torch.class('nn.DSSM_MMI_Criterion', 'nn.Criterion')

local Calculate_Alpha = require 'Calculate_Alpha'

function DSSM_MMI_Criterion:__init(batch_size, nTrail, gamma)
    parent.__init(self)
    self.batch_size = batch_size
    self.nTrail = nTrail
    self.gamma = gamma
    -- do negative sampling
    local negtive_array = torch.IntTensor(batch_size * nTrail)

    for i = 1, nTrail do
        local randpos = torch.random(0.8*batch_size) + math.floor(0.1 * batch_size)
        for k = 1, batch_size do
            local bs = (randpos + k) % batch_size + 1
            negtive_array[(i-1)*batch_size + k] = bs
        end
    end

    -- concatinate with the postive.
    self.D_negtive_array = negtive_array
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function DSSM_MMI_Criterion:updateOutput(input)
    
    local Q_input, D_input = input[1],input[2]
    local dimension = Q_input:size()[2]

    self.alpha_buffer = torch.Tensor(self.batch_size * (self.nTrail+1)):zero()
    self.Q_derivative_buffer = torch.Tensor(self.nTrail+1, self.batch_size, dimension):zero()
    self.D_derivative_buffer = torch.Tensor(self.nTrail+1, self.batch_size, dimension):zero()

    -- 1: calculate the cosine distance for positive and negtive array
    local CosDist = nn.CosineDistance()

    self.alpha_buffer:sub(1, self.batch_size):copy(CosDist:forward({Q_input, D_input}))
    local tmp_derivative = CosDist:backward({Q_input, D_input}, torch.Tensor(self.batch_size):fill(1))
    self.Q_derivative_buffer:sub(1,1):copy(tmp_derivative[1])
    self.D_derivative_buffer:sub(1,1):copy(tmp_derivative[2])

    -- for negtive sampling
    for i = 1,self.nTrail do
        -- shuffle the input index based on negative sampling.
        local D_input_neg = torch.Tensor(D_input:size()):zero()

        -- construct the negative input
        for j = 1,self.batch_size do
            D_input_neg:select(1,j):copy(D_input:select(1,self.D_negtive_array[(i-1)*self.batch_size + j]))
        end 

        self.alpha_buffer:sub(i * self.batch_size+1, (i+1)*self.batch_size):copy(CosDist:forward({Q_input, D_input_neg}))
        
        local tmp_derivative = CosDist:backward({Q_input, D_input_neg}, torch.Tensor(self.batch_size):fill(1))
        self.Q_derivative_buffer:sub(i+1,i+1):copy(tmp_derivative[1])
        self.D_derivative_buffer:sub(i+1,i+1):copy(tmp_derivative[2])
    end

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

function DSSM_MMI_Criterion:updateGradInput(input)
    -- here, we didn't use the fomula in the paper, we jsut
    -- j is the number of negative sampling
    -- gradInput = Sum_j[alpha_buffer[j] + (delta_postivePair - delta_negtivePair)]





end
