require 'nngraph'    
require 'nn'
Q_feature_size = 100
wind_size = 3
batch_size = 1024
    -- input is feature_size * window_size, filter is the same size. if we use the 
    -- 3D tensor, then the [dw] == 1
    
    Q_1_layer = nn.TemporalConvolution(Q_feature_size * wind_size, 1000, 1)()
    Q_1_link = nn.Tanh()(Q_1_layer)
    -- then a max pooling layer
    Q_1_pool = nn.TemporalMaxPooling(20)(Q_1_link)

    -- second layer
    Q_1_reshape = nn.Reshape(batch_size, 1000)(Q_1_pool)
    Q_2_layer = nn.Linear(1000, 300)(Q_1_reshape)
    --Q_2_link = nn.Tanh()(Q_2_layer)

    model = nn.gModule({Q_1_layer}, {Q_2_layer})


a = torch.Tensor(1024, 20, 300):random()


b = model:forward(a)

print(b:size())