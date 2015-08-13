local SparseTemporalConvolution, parent = torch.class('nn.SparseTemporalConvolution', 'nn.Module')

function SparseTemporalConvolution(inputFrameSize, outputFrameSize, kW)