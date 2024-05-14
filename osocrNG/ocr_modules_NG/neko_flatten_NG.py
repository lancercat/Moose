import torch

class neko_flatten_NG_lenpred:
    @classmethod
    def inflate(cls,out_emb,out_length):
        nB,nT=out_emb.shape[:2];
        start = 0;
        oucshp=[int(out_length.sum())]+list(out_emb.shape[2:]);
        output = torch.zeros(oucshp).type_as(out_emb.data)
        for i in range(0, nB):
            cur_length = int(out_length[i])
            cur_length_=cur_length
            if(cur_length_>nT):
                cur_length_=nT;
            output[start: start + cur_length_] = out_emb[i,0: cur_length_]
            # if(scores[cur_length_-1, i, :].argmax().item()!=0):
            #     print("???")
            start += cur_length_
        return output,out_length;
