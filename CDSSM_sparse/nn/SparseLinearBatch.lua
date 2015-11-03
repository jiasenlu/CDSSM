local SparseLinearBatch, parent = torch.class('nn.SparseLinearBatch', 'nn.Module')

function SparseLinearBatch:__init(inputSize, outputSize)
   parent.__init(self)

   self.weightDecay = 0
   self.weight = torch.Tensor(outputSize, inputSize):zero()
   self.bias = torch.Tensor(outputSize):zero()
   self.gradWeight = torch.Tensor(outputSize, inputSize):zero()
   self.gradBias = torch.Tensor(outputSize):zero()
   self.lastInput = nil

   self.gradInput:resize(inputSize)

   self:reset()
end

function SparseLinearBatch:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv) * 0.000001
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv):mul(0.000001)
   end
end

function SparseLinearBatch:updateOutput(input)
   self.batchSize = torch.max(input:select(2,1),1)[1]
   local outputSize = self.bias:size(1)
   if torch.getnumthreads() > 1 and outputSize >= 128 then
     self.shardBuffer = torch.Tensor(self.batchSize, outputSize, torch.getnumthreads())
   end
   self.output:resize(self.batchSize, outputSize):zero()

   return input.nn.SparseLinearBatch_updateOutput(self, input)
end

function SparseLinearBatch:accGradParameters(input, gradOutput, scale)
   if not self.lastInput then
     self.lastInput = input:clone()
   else
     self.lastInput:resizeAs(input):copy(input)
   end

   return input.nn.SparseLinearBatch_accGradParameters(self, input, gradOutput, scale)
end

function SparseLinearBatch:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nn.SparseLinearBatch_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end

function SparseLinearBatch:updateParameters(learningRate)
   self.lastInput.nn.SparseLinearBatch_updateParameters(self, learningRate)
end
