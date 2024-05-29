import torch
import vocab
from config import *
import numpy



class UsableTransformer:
    def __init__(self, model_path, vocab_path):
        print("Loading Vocab", vocab_path)
        self.vocab = vocab.WordVocab.load_vocab(vocab_path)
        print("Vocab Size: ", len(self.vocab))
        self.model = torch.load(model_path)
        self.model.eval()
        if USE_CUDA:
            self.model.cuda(CUDA_DEVICE)
    
    def encode(self, text, outputoption='lst'):

        segment_lable = []
        sequence = []
        for t in text:
            l = (len(t.split(' '))+2) * [1]
            s = self.vocab.to_seq(t)

            s = [3] + s + [2]
            if len(l) > 20:
                segment_lable.append(l[:20])
            else:
                segment_lable.append(l + [0]*(20-len(l)))
            
            if len(s) > 20:
                sequence.append(s[:20])
            else:
                sequence.append(s + [0]*(20-len(s)))

        #for 2023 paper fix length of function
        while len(sequence) < 350:
            sequence.append([0]*20)
            
        while len(segment_lable) < 350:
            segment_lable.append([0]*20)
        #########################################

        segment_lable = torch.LongTensor(segment_lable)
        sequence = torch.LongTensor(sequence)

        if USE_CUDA:
            sequence = sequence.cuda(CUDA_DEVICE)
            segment_lable = segment_lable.cuda(CUDA_DEVICE)

        encoded = self.model.forward(sequence, segment_lable)
        result = torch.mean(encoded.detach(), dim=1)

        del encoded

        if USE_CUDA:
            if numpy:
                return result.data.cpu().numpy()
            else:
                return result.to('cpu')

        else:
            if numpy:
                return result.data.numpy()
            else:
                return result

