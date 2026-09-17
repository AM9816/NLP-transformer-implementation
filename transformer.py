import torch
from torch import nn, Tensor as T
from torch.nn import functional
from torch.nn import Module

from tqdm import tqdm


from torch_common import *
from text_dataset import TextDataset

EPS = 1e-8

class MultiHeadSelfAttention(Module):
    def __init__(self, heads = 8, dim=128, dropout = 0.,
                 causal = True):
        super().__init__()

        assert dim % heads == 0, "head count must divide dim"

        self.causal = causal
        self.dim = dim
        self.headN = heads
        self.headDim = dim // heads

        # fuse query / key / value matrices for better performance
        self._fused_in = nn.Linear(dim, 3 * dim)
        self._fused_out= nn.Linear(dim, dim)
        self.p = dropout

    def forward(self, x):
        bsize, seqLength, _ = x.shape

        # get query, key and value matrices using fused linear operation
        # then chunk to split into seperate tensors
        _Q, _K, _V = self._fused_in(x).chunk(3, dim=-1)

        # optimised similarity measure
        _attention = functional.scaled_dot_product_attention(

            _Q.view(bsize, seqLength, self.headN, self.headDim)\
                .transpose(1, 2), 
            _K.view(bsize, seqLength, self.headN, self.headDim)\
                .transpose(1, 2), 
            _V.view(bsize, seqLength, self.headN, self.headDim)\
                .transpose(1, 2), 

            dropout_p=self.p if self.training else 0.,
            is_causal=self.causal
        )

        # resize and project out with fused linear operation
        _result = self._fused_out(
            _attention.transpose(1, 2).contiguous()
                .view(bsize, seqLength, self.dim))

        return _result


class TransformerMLP(nn.Sequential):
    def __init__(self, dim = 128, hiddenDim = 128 * 4, 
                 activ = nn.GELU(), dropout = 0):
        super().__init__(
            nn.Linear(dim, hiddenDim),
            activ, nn.Dropout(dropout),
            nn.Linear(hiddenDim, dim),
            nn.Dropout(dropout))


class TransformerBlock(Module):
    def __init__(self, dim = 128, heads = 8, mlpDim = 256, 
                 activ = nn.GELU(), dropout = 0.):
        super().__init__()

        self.norm1, self.norm2 = \
            nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.att = MultiHeadSelfAttention(heads, dim, dropout)
        self.mlp = TransformerMLP(dim, mlpDim, activ, dropout)        

    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.mlp(self.norm2(x)) 
        return x


class Transformer(nn.Module):
    def __init__(self, vocabSize, blocks = 6, maxSeqLen = 256,
                 dim = 256, heads = 8, mlpDim = 1024, 
                 emb = nn.Embedding, activ = nn.GELU(), 
                 tieVocabEmb = True,
                 dropout = 0.1):
        super().__init__()

        self.vocabEmb = emb(vocabSize, dim)
        self.posEmb = emb(maxSeqLen, dim)
        self.transformers = nn.Sequential(*[
            TransformerBlock(dim, heads,
                             mlpDim, activ, dropout)
                for _ in range(blocks)
        ])

        # if using a single weight for embedding and final
        # projection, skip bias to lower param count
        self.out = nn.Linear(dim, vocabSize,
                             bias = not tieVocabEmb)
        if tieVocabEmb:
            self.out.weight = self.vocabEmb.weight


    def forward(self, x) -> Tensor:
        _, size = x.shape
        x = self.vocabEmb(x) + self.posEmb(
            torch.arange(0, size, device=x.device))
        x = self.transformers(x)
        x = self.out(x)
        return x

    
    def generate(self, start, dataset, 
                 maxLength = 1024, temperature = 1.,
                 generator = True, forceMaxLength = False,
                 device=gpu):
        
        with torch.no_grad():

            self = self.eval().to(device)
            id_token = {v: k for k, v in dataset.vocab.items()}
            unknown = dataset.unknown
            unknownId = dataset.vocab.get(unknown, 0)
            literalUnkId = dataset.vocab.get("<unk>")
            eos = dataset.eos
            eosId = dataset.vocab.get(eos, 1)

            tokens = start.split()
            ids = [
                dataset.vocab.get(tok, unknownId) for tok in tokens
            ] if tokens else [unknownId]

            if unknownId in ids:
                print("start text has unknown token, "
                    f"{[id_token[_id] for _id in ids]}")

            out = torch.tensor([ids], dtype=torch.long, device=device)
            # outStr = ''

            # print(out.shape); exit()

            if generator:
                yield start

            for _ in range(maxLength):
                logits = self(out)
                nextLogits = logits[:, -1, ...] / max(temperature, EPS)

                # mask out unknown tokens
                nextLogits[:, unknownId] = float('-inf')

                if literalUnkId is not None:
                    nextLogits[:, literalUnkId] = float('-inf')

                if forceMaxLength:
                    nextLogits[:, eosId] = float('-inf')

                if temperature > 0:
                    dist = functional.softmax(nextLogits, dim=-1)
                    nextId = torch.multinomial(dist, num_samples=1)
                else:
                    nextId = torch.argmax(nextLogits, dim=-1, keepdim=True)

                out = torch.cat([out, nextId], dim = 1)

                if generator:
                    # yield ' '.join([
                    #     id_token.get(_id.item(), unknown) for _id in out[0]
                    # ])
                    yield f'{id_token[nextId.item()]} '

                if nextId.item() == eosId:
                    break


            if generator:
                yield "\n"

            return ' '.join([
                id_token.get(_id.item(), unknown) for _id in out[0]])



def train_transformer(model: Transformer, dataset: TextDataset, 
         bsize = 32, lr = 1e-3, epochs = 10, trainTestSplit = .25,
         doCompile = True, useGradScaler = True, scalerPrecision = torch.float16,
         device=gpu):
         
    # dataset = TextDataset(path, seqLength)
    _train, _test = get_train_test_split(dataset, trainTestSplit, bsize, True)

    dtype = next(model.parameters()).dtype
    vocabSize = dataset.vocabSize()

    model : nn.Module = model.train().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss(
        ignore_index=dataset.vocab[dataset.unknown],
        label_smoothing=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=lr, total_steps=epochs * len(_train))
    scaler = torch.amp.grad_scaler.GradScaler(device.type)\
             if useGradScaler else None



    if doCompile:
        print("compiling model .. ", end='')
        model = torch.compile(model)
        print("done")

    for epoch in range(1, epochs + 1):
        sumLoss = sumTestLoss = 0

        saveModel(model)

        with torch.no_grad():
            for word in model.generate("hello ", dataset, 20):
                print(word, end="")

        model.train()

        _iter = enumerate(_train)
        for i, (X, Y) in (pb:=tqdm(_iter, total=len(_train))):

            X, Y = X.to(device), Y.to(device)
            optim.zero_grad()

            with torch.autocast(device_type=device.type, dtype=scalerPrecision):
                out = model(X)
                loss = lossf(out.view(-1, vocabSize), Y.view(-1))

            # loss.backward()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            scaler.update()

            sumLoss += loss.item()

            pb.set_description_str(f"{sumLoss / (i+1)}")

        nTrain = i + 1

        _iter = enumerate(_test)
        with torch.no_grad():
            for i, (X, Y) in (pb:=tqdm(_iter, total=len(_test))):
                X, Y = X.to(device), Y.to(device)
                # Y = nn.functional.one_hot(Y, vocabSize).to(dtype)
                out = model(X)
                loss = lossf(out.view(-1, vocabSize), Y.view(-1))
                sumTestLoss += loss.item()



        avgLoss = sumLoss / nTrain
        avgTestLoss = sumTestLoss / (i+1)

        print(f"epoch {epoch} done, loss = {avgLoss}, v loss = {avgTestLoss}")

    return model



if __name__ == '__main__':

    bsize = 32
    lr = 3e-4
    seql = 256
    stride = 128
    ttsplit = .1
    

    # # for f in range(10):

    dataset = TextDataset('wikitext.txt', seql, stride,
                        minTokenFreq=6)
    print(f'vocab size = {dataset.vocabSize()}')
    

    model = Transformer(dataset.vocabSize(),
                        dropout = .1)
    # # model = loadModel()
    # print(model)

    print("effective max matrix size = "
          f"{bsize * dataset.vocabSize()}")
    
    train_transformer(model, dataset, bsize, lr, 20,
                      doCompile=False,
                      trainTestSplit=ttsplit)



    # model:Transformer = loadModel(r"C:\\Users\\artur\\Downloads\\latest (25).pt")
    # print(model)

    # for word in model.generate("josh likes ", dataset, int(1e2), .5, 
    #                            forceMaxLength=True):
    #     print(word, end = '')

