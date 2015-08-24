
local Calculate_Alpha = {}
Calculate_Alpha.__index = Calculate_Alpha


function Calculate_Alpha.cal_alpha(alpha, nTrial, batchsize, gamma)
    -- alpha[negtive] = exp(-gamma * (alpha[positive] - alpha[negative])
    local positive_array = torch.range(1,batchsize):type('torch.IntTensor')  
    local pos_alpha = alpha:sub(1,batchsize)
    local neg_alpha = alpha:sub(batchsize+1, -1)
    positive_array = positive_array:repeatTensor(nTrial) -- replicate the nTrail times
    positive_array = positive_array:type('torch.LongTensor')

    local pos_alpha_replicate = pos_alpha:index(1,positive_array)

    neg_alpha = torch.add(pos_alpha_replicate, - neg_alpha) 
    neg_alpha = torch.exp(torch.mul(neg_alpha, gamma))
    
    local new_alpha = torch.cat(pos_alpha, neg_alpha)
    return new_alpha
end

function Calculate_Alpha.cal_alpha_sum(alpha, nTrial, batchsize, gamma, init)
    -- alpha[postive] = init + alpha[postive + negative] (sum of all the (pos - negtive))
    local pos_alpha = alpha:sub(1,batchsize)
    local neg_alpha = alpha:sub(batchsize+1, -1)    
    local neg_alpha_split = neg_alpha:split(batchsize, 1)
    for i = 1, nTrial do
      pos_alpha:add(neg_alpha_split[i])
    end
    pos_alpha:add(init)
    local new_alpha = torch.cat(pos_alpha, neg_alpha)
    return new_alpha
end

function Calculate_Alpha.cal_alpha_norm(alpha, nTrial, batchsize, gamma)
    -- alpha[negative] = gamma * alpha[negative]./ alpha[positive] (normalization)
    local pos_alpha = alpha:sub(1,batchsize)
    local neg_alpha = alpha:sub(batchsize+1, -1)
    local positive_array = torch.range(1,batchsize):type('torch.IntTensor')  
    positive_array = positive_array:repeatTensor(nTrial) -- replicate the nTrail times
    positive_array = positive_array:type('torch.LongTensor')
    local pos_alpha_replicate = pos_alpha:index(1,positive_array)

    neg_alpha = torch.cdiv(neg_alpha, pos_alpha_replicate)
    neg_alpha = torch.mul(neg_alpha, gamma)
    local new_alpha = torch.cat(pos_alpha, neg_alpha)

    return new_alpha
end

function Calculate_Alpha.cal_derivative(derivative)
    local pos_derivative = derivative:sub(1,1)
    pos_derivative = pos_derivative:expandAs(derivative)
    derivative = -derivative
    derivative:add(pos_derivative)

    return derivative

end
return Calculate_Alpha