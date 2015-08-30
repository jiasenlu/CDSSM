local SparseConvolution, parent = torch.class('nn.SparseConvolution', 'nn.Module')

function SparseConvolution:__init(featsize, N_Winsize, outputSize)
    parent.__init(self)

    local inputSize = featsize * N_Winsize

    self.N_Winsize = N_Winsize
    self.weight = torch.Tensor(outputSize, inputSize):zero()
    self.bias = torch.Tensor(outputSize):zero()
    self.gradWeight = torch.Tensor(outputSize, inputSize):zero()
    self.gradBias = torch.Tensor(outputSize):zero()

    self.lastInpyt = nil

   if torch.getnumthreads() > 1 and outputSize >= 128 then
     self.shardBuffer = torch.Tensor(outputSize, torch.getnumthreads())
   end

   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)

   self:reset(inputSize, outputSize)

end

function SparseConvolution:reset(inputSize, outputSize)
    stdv = math.sqrt(6 / (inputSize + outputSize))*2
    bias = -math.sqrt(6 / (inputSize + outputSize))

    self.weight:uniform(-stdv, stdv):add(bias)
    self.bias:zero()
end

function SparseConvolution:updateOutput(input)
    return input.nn.SparseConvolution_updateOutput(self, input)
end


