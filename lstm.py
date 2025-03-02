# Ahmet SOYYİĞİT
# The code is working but performance is not so good,
# please inform me if you have any suggestions
# lstm inputs are 3D, [sequence,batch_num,data]
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_utils as du
import numpy as np

torch.set_default_device('cuda')
torch.manual_seed(1)

print(torch.cuda.is_available())
print(torch.backends.cudnn.version())


# [timesteps, numOfSequences(number of batch elements), inputSize]
trainDir = "./MSRAction3DSkeletonReal3D"
trainData, trainLabels = du.read_msrdataset(trainDir,40)
tLEn = trainData.shape[1]
p = np.random.permutation(trainData.shape[1])
trainData, trainLabels = trainData[:,p,:],trainLabels[p]
testData, testLabels = trainData[:,(tLEn-tLEn//5):,:],trainLabels[(tLEn-tLEn//5):]
trainData, trainLabels = trainData[:,:(tLEn-tLEn//5),:],trainLabels[:(tLEn-tLEn//5)]
print(trainData.shape)
print(testData.shape)
print(trainLabels.shape)
print(testLabels.shape)

class denseLSTM(nn.Module):

    def __init__(self):
        super(denseLSTM, self).__init__()
        
        #self.fc_init_layer = nn.Linear(60,12, bias=False)
        self.l1_LSTM  = nn.LSTM(60,64)
        #There is a dropout here
        self.l2_LSTM  = nn.LSTM(60+64,128)
        #There is a dropout here
        self.l3_LSTM  = nn.LSTM(60+64+128,256)
        #There is a dropout here
        self.fc_last_layer  = nn.Linear(256,20)

    def init_hidden(self,batchSize):
        self.hidden = [
            (
                autograd.Variable(torch.zeros(1, batchSize, hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batchSize, hidden_dim).cuda())
            ) for hidden_dim in [64,128,256]
        ]
    
    
    def forward(self, inpBatch):
        self.init_hidden(inpBatch.size()[1])
        
        # Forward input through network        
        l1_lstm_out, self.hidden[0] = self.l1_LSTM(
                                                    inpBatch,
                                                    self.hidden[0]
                                                   )        
        
        l2_lstm_out, self.hidden[1] = self.l2_LSTM(
                                                    F.dropout(torch.cat( (inpBatch,l1_lstm_out), 2 ),0.95),
                                                    self.hidden[1]
                                                   )        
        
        l3_lstm_out, self.hidden[2] = self.l3_LSTM(
                                                    F.dropout(torch.cat((inpBatch,l1_lstm_out,l2_lstm_out),2),0.95),
                                                    self.hidden[2]
                                                   )
        

        
        fc_linear_out =self.fc_last_layer(F.dropout(l3_lstm_out[-1],0.95))
        return fc_linear_out
    
    #create network object
model = denseLSTM().cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
torch.cuda.empty_cache()

def dataBatchLens(inputData,batchSize):
    l = inputData.shape[1]
    bLens = [ [b,b+batchSize] for b in range(0,l-l%batchSize,batchSize) ]
    if l-l%batchSize != l:
        bLens.append([l-l%batchSize, l])
    return bLens
trainBatchLens = dataBatchLens(trainData,64) 
testBatchLens = dataBatchLens(testData,64) 

#print(trainBatchLens)
#print(testBatchLens)


f = open('lstm.txt','w')
for epoch in range(300):
    correctPredictionsTr = 0
    totalLossTr = 0.0
    #Shuffle data before each epoch
    p = np.random.permutation(trainData.shape[1])
    trainData, trainLabels = trainData[:,p,:],trainLabels[p]
    for bLen in trainBatchLens:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        batch = autograd.Variable(torch.from_numpy(trainData[:,bLen[0]:bLen[1]]).cuda())
        labels = autograd.Variable(torch.from_numpy(trainLabels[bLen[0]:bLen[1]]).cuda())

        # Step 3. Run our forward pass.
        results = model(batch)
        _ , idx = torch.max(torch.exp(results),1)
        correctPredictionsTr += torch.sum(torch.eq(idx,labels)).item()
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(results, labels)
        totalLossTr += loss.item()
        loss.backward()
        optimizer.step()
    
    
    # After each epoch, test
    correctPredictionsTe = 0
    totalLossTe = 0.0
    for bLen in testBatchLens:
        batch = autograd.Variable(torch.from_numpy(testData[:,bLen[0]:bLen[1]]).cuda())
        labels = autograd.Variable(torch.from_numpy(testLabels[bLen[0]:bLen[1]]).cuda())
        results = model(batch)
        _ , idx = torch.max(torch.exp(results),1)
        correctPredictionsTe += torch.sum(torch.eq(idx,labels)).item()
        loss = loss_function(results,labels)
        totalLossTe += loss.item()
    print("Epoch " + str(epoch+1)+",Train loss    : " + str(totalLossTr / trainLabels.shape[0]))
    print("Epoch " + str(epoch+1)+",Test  loss    : " + str(totalLossTe / testLabels.shape[0]))
    print("Epoch " + str(epoch+1)+",Train accuracy: " + str(correctPredictionsTr) + "/" + 
          str(trainLabels.shape[0]) + " %" + str(int(correctPredictionsTr/trainLabels.shape[0]*100)))
        
    acc = round(correctPredictionsTe/testLabels.shape[0]*100,3)
    
    f.write('{:.3f} {:.3f}\n'.format(acc,loss.item()))
    print("Epoch " + str(epoch+1)+",Test  accuracy: " + str(correctPredictionsTe) + "/" + 
          str(testLabels.shape[0]) + " %" + str(acc) + str(loss.item()))

f.close()

    