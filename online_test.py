import json
import random
import torch
from tqdm import tqdm
import argparse
import numpy as np
import os, pickle

# import sys
# sys.path.append("/home/zkx/ImgCap/CaDReL/captioning/common/")

from common.utils.utils import setup_seed
from common.data import OnlineTestDataset, DataLoader
from common.data.field import ImageDetectionsField, RawField, TextField, DualImageField
# from common.models.transformer import Transformer
from DualModel.models.transformer import Transformer, TransformerEnsemble
# from DualModel.models import build_encoder, build_decoder
# from AttModel.models import build_encoder, build_decoder
from Camel.models.captioner import Captioner


# setup_seed(1234)
setup_seed(3407)
torch.backends.cudnn.benchmark = True

def online_test(model, dataloader, text_field, device):
    import itertools
    model.eval()
    result = []
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, image_ids) in enumerate(iter(dataloader)):
            with torch.no_grad():
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)
                
                # 1
                out, _ = model.beam_search(images, 30, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            
            caps_gen = text_field.decode(out, join_words=False)
            for i, (image_id, gen_i) in enumerate(zip(image_ids, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                result.append({"image_id": image_id.item(), "caption": gen_i.strip()})
            pbar.update()
    return result

def test():
    parser = argparse.ArgumentParser(description='CaMEL')
    parser.add_argument('--output', type=str, default='Camel')
    parser.add_argument('--exp_name', type=str, default='Camel')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--clip_variant', type=str, default='RN50x16')
    parser.add_argument('--distillation_weight', type=float, default=0.1)
    parser.add_argument('--ema_weight', type=float, default=0.999)
    parser.add_argument('--enable_mesh', action='store_true')
    parser.add_argument('--use_premlp', action='store_true')

    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--d_in', type=int, default=3072)
    # RN50x4 = 2560  RN50x16 = 3072 
    # parser.add_argument('--model_path', type=str, default='/home/zkx/ImgCap/CaDReL/captioning/Camel/saved/141.11.pth')
    parser.add_argument('--feature_type', type=str, default='clip')
    # 2
    parser.add_argument('--features_path', type=str, default='/home/public/caption/caption/coco2014/feats/COCO2014_RN50x16_GLOBAL.hdf5')
    parser.add_argument('--image_folder', type=str, default='/home/public/caption/caption/coco2014/images')
    parser.add_argument('--annotation_folder', type=str, default='/home/public/caption/caption/coco2014/annotations')
    args = parser.parse_args()

    print('DRL Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(feature_type=args.feature_type, detections_path=args.features_path, max_detections=50)
    # image_field = DualImageField(max_detections=50, global_feature=False, online_test=True)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    vocab_path = '/home/zkx/ImgCap/Discrimitor/captioning/cache/vocab.pkl'
    text_field.vocab = pickle.load(open(vocab_path, 'rb'))

    # 3
    # ann_path = '/home/public/caption/caption/coco2014/annotations/image_info_test2014.json'
    ann_path = '/home/public/caption/caption/coco2014/annotations/captions_val2014.json'
    dataset_test = OnlineTestDataset(ann_path, {'image': image_field, 'image_id': RawField()})
    dict_dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=False)

    # Model and dataloaders
    # encoder = build_encoder(3, device=args.device)
    # decoder = build_decoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])

    
    online_model = Captioner(args, text_field, istarget=1).to(args.device)
    # online_data = torch.load('/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/141.11.pth',map_location= args.device)
    # online_data = torch.load('/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/base_119.pth',map_location= args.device)
    # online_model.load_state_dict(online_data['state_dict_t'])
    # online_model.load_state_dict(online_data['state_dict_o'])

    # model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder)
    weight_files = ['/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/141.11.pth',
                    '/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/141.0.pth', 
                    '/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/140.9.pth', 
                    '/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/140.7.pth']
    online_model = TransformerEnsemble(online_model, weight_files, args.device).to(args.device)


    online_model.to(args.device)
    result = online_test(online_model, dict_dataloader_test, text_field, args.device)

    # 4
    # output_path = '/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/result/captions_test2014_DRL_results_5e3407.json'
    output_path = '/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/result/captions_val2014_DRL_results_5e3407.json'
    with open(output_path, 'w') as fp:
        json.dump(result, fp)

    print("Online test result is over")

test()