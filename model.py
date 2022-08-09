import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.e = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.e(captions[:,:-1])
        
        inputs = torch.cat((features.unsqueeze(dim=1),embeddings), dim=1)
        
        lstm_out, _ = self.lstm(inputs)
        
        outputs = self.linear(lstm_out)
        
        return outputs
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        out = list()
        
        for i in range(max_len):
            probabilities, states = self.lstm(inputs, states)
            
            probabilities = self.linear(probabilities)
            
            probabilities = probabilities.cpu().detach().numpy()[0][0].tolist()
            
            word = probabilities.index(max(probabilities))
            
            out.append(word)
            
            if word == 1:
                break
   
            inputs = self.e(torch.LongTensor([word]).unsqueeze(1).to(inputs.device))

                
               
        return out       
            

               
            
        
        
        
        
        
        
        
        