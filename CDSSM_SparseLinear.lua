local CDSSM_SparseLinear, parent = torch.class('nn.CDSSM_SparseLinear', 'nn.Module')


function CDSSM_SparseLinear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weightDecay = 0
   self.weight = torch.Tensor(outputSize, inputSize):zero()
   self.bias = torch.Tensor(outputSize):zero()
   self.gradWeight = torch.Tensor(outputSize, inputSize):zero()
   self.gradBias = torch.Tensor(outputSize):zero()
   self.lastInput = nil


   self:reset(inputSize, outputSize)
end

function CDSSM_SparseLinear:reset(inputSize, outputSize)
    stdv = math.sqrt(6 / (inputSize + outputSize))*2
    bias = -math.sqrt(6 / (inputSize + outputSize))

    self.weight:uniform(0, stdv):add(bias)
    self.bias:zero()
end

function CDSSM_SparseLinear:updateOutput(input)
    local Smp_Index = input.sample_Idx_Mem
    local Seg_Index = input.seg_Idx_Mem
    local Seg_Margin = input.seg_Margin_Mem
    local Seg_Len = input.seg_Len_Mem
    local Fea_Index = input.fea_Idx_Mem
    local Fea_Value = input.fea_Value_Mem
    local batchsize = input.batch_size

    local inputDimension = self.weight:size()[2]
    local outputDimension = self.weight:size()[1]
    local output = torch.Tensor(batchsize, outputDimension)

    local total = batchsize * outputDimension

    for id = 1, total do
        local batch_idx = math.ceil(id / outputDimension)
        local output_idx = (id-1) % outputDimension+1

        local seg_end = Smp_Index[batch_idx]
        local seg_begin = 0
        if batch_idx > 1 then
            seg_begin = Smp_Index[batch_idx-1]
        end

        local sum = 0

        for word_idx = seg_begin+1, seg_end do
            local col_end = Seg_Index[word_idx]
            local col_begin = 0
            if word_idx > 1 then
                col_begin = Seg_Index[word_idx-1]
            end

            for i = col_begin+1, col_end do
                local fea_idx = Fea_Index[i]
                sum = sum + Fea_Value[i] * self.weight[output_idx][(word_idx - seg_begin-1) * inputDimension + fea_idx]
            end
        end
        output[batch_idx][output_idx] = sum
    end
    return output
end
