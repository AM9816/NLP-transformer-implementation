import torch

from collections import Counter
import os, re
from tqdm import tqdm
from xxhash import xxh3_128_hexdigest as fasthash

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path, seqLength = 128, stride=1,
                 minTokenFreq = 2, unknownToken='<UNKNOWN>',
                 eosToken = '<EOS>', cacheDataset = False,
                 charLevel = False):

        cachePath = f"{path}.pt" if cacheDataset else None

        self.stride = stride
        self.seqLength = seqLength
        self.isChLevel = charLevel

        if cacheDataset and os.path.exists(cachePath):
            try:
                self.loadCache(cachePath)
                assert self.ids.max() < len(self.vocab)
                return
            except Exception as e:
                print(f"cache load failed, {e}\n"
                       "reconstructing .. ")

        try:
            text = self.loadText(path)
        except UnicodeDecodeError as e:
            text = self.loadText(path, encoding='utf-8')
        except Exception as e:
            print(f'file read exception, {e}')
            return
        

        if eosToken is not None:
            text = text.replace("\n\n", f" {eosToken} ")

        text = text.lower().replace('<unk>', '') 

        # char vs word level tokenization        
        if charLevel:
            tokens = re.findall(r"<[^\s>]+>|[\s\S]", text)
        else:
            tokens = re.findall(r"<[^\s>]+>|\w+", text)

        self.unknown = unknownToken        
        self.eos = eosToken

        self.vocab = {self.unknown: 0}

        if eosToken is not None:
            self.vocab[eosToken] = 1

        counts = Counter(tokens)
        tokensUsed = [w for w, c in counts.items() 
                      if c >= minTokenFreq and w not in self.vocab]
        tokensUsed.sort(key=lambda word: (-counts[word], word))

        # print([w for w, c in counts.items() 
        #               if c < minTokenFreq and w not in self.vocab])

        # allocate each used token a unique id
        i = len(self.vocab)
        for token in tqdm(tokensUsed):
            self.vocab[token] = i
            i += 1

        unknownId = self.vocab[self.unknown]
        self.ids = [self.vocab.get(token, unknownId) for token in tqdm(tokens)]
        self.data = torch.tensor(self.ids, dtype=torch.long)

        if cacheDataset:
            # os.makedirs(cachePath, exist_ok=True)
            self.saveCache(cachePath)

    def loadText(self, path, encoding=None):
        with open(path, "r", encoding=encoding) as f:
            return f.read()
        

    def loadCache(self, path):
        cached = torch.load(path, weights_only=False)
        self.vocab = cached["vocab"]
        self.data = cached["data"]
        self.ids = cached["ids"]
        # self.seqLength = cached["sl"]
        # self.stride = cached["st"]
        self.unknown = cached["uk"]
        self.eos = cached["eos"]

    def saveCache(self, path):
        torch.save(
            {   "ids": self.ids, 
                # "sl": self.seqLength, 
                "uk":self.unknown, 
                # "st": self.stride, 
                "eos": self.eos, "vocab": self.vocab, 
                "data": self.data
            }, path)

    def vocabSize(self):
        return len(self.vocab)

    def __len__(self):
        n = len(self.data) - self.seqLength - 1
        if n <= 0:
            return 0
        return (n // self.stride) + 1

    def __getitem__(self, index):
        start = index * self.stride
        chunk = self.data[start : start + self.seqLength + 1]
        return chunk[:-1], chunk[1:]