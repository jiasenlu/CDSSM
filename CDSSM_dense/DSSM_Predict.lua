local DSSM_Predict = {}
DSSM_Predict.__index = DSSM_Predict

function DSSM_Predict:updateOutput(input)

    local Q_input, D_input = input[1], input[2]
    
    local Q_len = Q_input:size(1)
    local D_len = D_input:size(1)
    
    local prob_matrix = torch.Tensor(Q_len, D_len):zero()

    self.input2 = D_input
    for i = 1, Q_len do
        self.input1 = Q_input:sub(i,i):expandAs(self.input2)
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

        self.dist = torch.cmul(self.w1,self.w)
        self.dist = self.dist:select(2,1)
        prob_matrix:sub(i,i):copy(self.dist)
    end

    -- find the max value
    local _, predict = torch.max(prob_matrix,2)
    local correct = 0
    for i = 1, D_len do
        local predict_label = label[predict[i][1]]
        if predict_label == label[i] then
            correct = correct + 1
        end
    end
    print('accuracy are: ' .. correct / D_len)
    --print(math.sum() / D_len) 
    return prob_matrix
end


return DSSM_Predict