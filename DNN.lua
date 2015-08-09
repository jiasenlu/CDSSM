-- build the neural network

require 'torch'
require 'nn'

function DNN.init()





end

function DNN.training()

end


function DNN.ModelInit_FromConfig(feature_size, layerDim, activation, sigma, arch, wind, backupOnly)

    -- initialzing the network

    -- N_windows.

    -- copy the initial paramters setting


    model = nn.Sequential()
    model:add(nn.Linear(feature_size, layerDim[1]))

    for i = 1, #layerDim do
        model:add()


    end



    -- if !Parame.is_seed -- create new model here.

end


-- function Neuralink 
-- Nt ?
-- N_Winsize ? (window size)

-- Af: function

-- initHidBias = hidBias?
-- initWeightSigma = weightSigma




function DNN.LoadTrainData(srcFile, tgtFile)

    -- Construct shuffleTrain file

    -- verify the train file and tgt file count the same

    -- if train file count == 0, error

    -- random select the train file and target file

    -- function (LoadPairdataAtIdx)

end


function DNN.LoadPairDataAtIdx()
    -- print load pair tain data
    -- if objective type == NCE
        -- load nceProbFile
        -- if contain shuffle then load shuffled nceProbfile (I guess)

    -- function (LoadTrainPairData(qFileName, dFileName, nceProbDisFile))


    -- Normalizer.CreateFeatureNomalizer() return the feature dim

    -- InitFeatureNorm(srcNormalizer, tgtNormalizer) (seems didn;t do anything here)

    -- pairTrainFileIdx = (pairTrainFileIdx + 1) % pairTrainFiles.count

end


function DNN.LoadTrainPairData(qFileName, dFileName, nceProbfile)
    -- function (LoadPairData())






end

function DNN.LoadPairData(qFileName, dFileName, nceProbDistFile)
    

    -- get dimension of qFile.

    -- QUERY_MAXSEGMENT_BATCH = 40000

    --QUERY_MAXSEGMENT_BATCH = match.max(QUERY_MAXSEGMENT_BATCH, dim.max)

    -- Doc_Maxsegment_batch

    -- MaxSegment_batch = max(Q, D)




end


function DNN.get_dimension(FileName)
    -- last five dimension

    -- function BatchSample_Input()
    
    -- initial the data

    -- batch_num = (total_batch_size / param.batch_size) + 1

    -- last_incomplete_batch_size = total_batch_size % param.batch_size

    -- batch_index = 0

end

function DNN.BatchSample_Input(max_batch_size, maxSequence_perBatch, maxElement_perBatch)




end

function DNN.ConstructShuffleTrainFiles(file)
    -- seems only shuffle the train file name if there are multiple file


end

return DNN