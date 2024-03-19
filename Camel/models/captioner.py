import torch
from torch import nn
import copy

from Camel.utils.beam_search import BeamSearch
from M2T.models.decoder import MeshedDecoder
from Camel.models.meshed_encoder import MemoryAugmentedEncoder
from common import utils
from common.data.field import TextField
from torch.nn import functional as F
from common.models.captioning_model import CaptioningModel
from common.models.containers import ModuleList
# from common.models.transformer import TransformerEncoder, TransformerDecoder 
from Camel.models.Diff_Att_encoders import TransformerEncoder as DiffusionAttentionTransformerEncoder
from Camel.models.decoders import TransformerDecoder
# from TargetEncoder import TargetEncoder

class Captioner(CaptioningModel):
    def __init__(self, args, text_field: TextField, istarget):
        super(Captioner, self).__init__()
        # self.encoder = MemoryAugmentedEncoder(args.N_enc, 0, args.m, d_in=args.image_dim, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1)
        # self.encoder = TransformerEncoder(args.N_enc, 0, d_in=args.d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1)
        if istarget==1:
            self.encoder = DiffusionAttentionTransformerEncoder(args.N_enc, 0, d_in=args.d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, iscamel=1, batch_size=args.batch_size, isMesh=args.enable_mesh)
            # self.encoder = TransformerEncoder(args.N_enc, 0, d_in=args.d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1)
            if args.enable_mesh:
                self.encoder = MemoryAugmentedEncoder(args.N_enc, 0, args.m, d_in=args.d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, iscamel=1, batch_size=args.batch_size, isMesh=args.enable_mesh)
        
        if args.enable_mesh:
            self.decoder = MeshedDecoder(len(text_field.vocab), 54, args.N_dec, text_field.vocab.stoi['<pad>'], 
                                         d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, return_logits=True, batch_size=args.batch_size)
        else:
            self.decoder = TransformerDecoder(len(text_field.vocab), 54, args.N_dec, text_field.vocab.stoi['<pad>'], 
                                              d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, return_logits=True, batch_size=args.batch_size)

        self.bos_idx = text_field.vocab.stoi['<bos>']
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        # /w\
        self.register_state('att_outs', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, images, seq, *args):
        if not isinstance(images, tuple) and not isinstance(images, list):
            images = [images]

        if args[0].enable_mesh:
            out, attention_mask, att_outs, outs = self.encoder(*images)

        else:    
            out, attention_mask, att_outs = self.encoder(*images)

        # if not isinstance(enc_output, tuple) and not isinstance(enc_output, list):
        #     enc_output = [enc_output]
        # enc_output = [out, attention_mask]
        # print(enc_output.shape)

        if args[0].enable_mesh:
            pred_out, dis_out = self.decoder(seq, out, attention_mask, outs)
        
        else:
            pred_out, dis_out = self.decoder(seq, out, attention_mask)
    
        # pred_out = self.decoder(seq, out, attention_mask)
        return pred_out, att_outs, dis_out, out

        # pred_out, multi_level_outs = self.decoder(seq, *enc_output)
        # return pred_out, multi_level_outs
        #[1,2,3,4]

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        # isEnableMesh = self.encoder.isMesh
        isEnableMesh = 0
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                if not isinstance(visual, tuple) and not isinstance(visual, list):
                    visual = [visual]

                # self.enc_output, self.mask_enc = self.encoder(*visual)
                # enc_output, mask_enc = self.encoder(*visual)
                enc_output = self.encoder(*visual)

                # /w\
                if isinstance(enc_output, tuple) or isinstance(enc_output, list):
                    if isEnableMesh:
                        self.enc_dif_output, self.mask_enc, self.enc_output = enc_output[0], enc_output[1], enc_output[3]
                    else:
                        self.enc_output, self.mask_enc = enc_output[0], enc_output[1]
                    # self.enc_output, self.mask_enc, self.att_outs = enc_output[0], enc_output[1], enc_output[2]
                else:
                    self.enc_dif_output, self.mask_enc = enc_output, None

                it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        if isEnableMesh:
            logits, _ = self.decoder(it, self.enc_dif_output, self.mask_enc, self.enc_output)
        else:
            logits, _ = self.decoder(it, self.enc_output, self.mask_enc)

        return logits
        # /w\
        # return logits

    def beam_search(self, visual: utils.TensorOrSequence, max_len: int, eos_idx: int, beam_size: int, out_size=1,
                    return_probs=False, **kwargs):
        bs = BeamSearch(self, max_len, eos_idx, beam_size)
        return bs.apply(visual, out_size, return_probs, **kwargs)


class CaptionerEnsemble(CaptioningModel):
    def __init__(self, model: Captioner, weight_files):
        super(CaptionerEnsemble, self).__init__()
        # self.n = len(weight_files)
        self.n = 2
        weight_file = 'Camel/saved_models/Camel_best.pth'
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        # for i in range(self.n):
            # state_dict_i = torch.load(weight_files[i])['state_dict']
            # self.models[i].load_state_dict(state_dict_i)
        state_dict = torch.load(weight_file)
        self.models[0].load_state_dict(state_dict['state_dict_t'])
        self.models[1].load_state_dict(state_dict['state_dict_o'])

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_i = F.log_softmax(out_i, dim=-1)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
