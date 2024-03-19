import torch
import argparse
from Camel.models.captioner import Captioner
from Camel.utils.train import train, train_s
from Camel.utils.test import test
# from common.visualization_base import test
from common.visualization import test
from ClipModel.utils.utils import create_transform
from common.data.field import ImageDetectionsField, ImageWithDAField, TextField
from common.models.backbone import clip
from common.utils.utils import create_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='CaDReL Model')
    parser.add_argument('--output', type=str, default='CaDReL')
    parser.add_argument('--exp_name', type=str, default='CaDReL')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=65)
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

    parser.add_argument('--xe_base_lr', type=float, default=1e-4)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--use_rl', action='store_true')
    parser.add_argument('--use_srl', action='store_true')
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--feature_type', type=str, default='clip')
    parser.add_argument('--features_path', type=str, default='/home/public/caption/caption/coco2014/feats/COCO2014_RN50x16_GLOBAL.hdf5')
    parser.add_argument('--image_folder', type=str, default='/home/public/caption/caption/coco2014/images')
    parser.add_argument('--annotation_folder', type=str, default='/home/public/caption/caption/coco2014/annotations')
    args = parser.parse_args()
    print(args)

    return args



def main(args):
    print('Camel Model Training')

    # Pipeline for image regions
    if args.online:
        clip_variant = 'RN50x4'
        clip_model, _ = clip.load(clip_variant, device=args.device, jit=False)
        image_model = clip_model.visual.to(args.device).eval()
        image_model.forward = image_model.intermediate_features
        train_transform, test_transform = create_transform(n_px=288)
        image_field = ImageWithDAField(train_transform, test_transform)
    else:
        image_model = None
        image_field = ImageDetectionsField(feature_type=args.feature_type, detections_path=args.features_path)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False, reverse=False)

    datasets = create_dataset(args, image_field, text_field)

    if args.use_srl:
        online_model = Captioner(args, text_field, istarget=1).to(args.device)
        online_data = torch.load('/home/zkx/ImgCap/CaDReL/captioning/Camel/saved/141.11.pth',map_location= args.device)
        online_model.load_state_dict(online_data['state_dict_t'])
        online_model.to(args.device)
        train_s(args, image_model, online_model, datasets, image_field, text_field)
        exit

    target_model = Captioner(args, text_field, istarget=1).to(args.device)
    online_model = Captioner(args, text_field, istarget=1).to(args.device)

    if args.use_rl:
        target_data = torch.load('/home/zkx/ImgCap/CaDReL/captioning/Camel/saved/129.72_2.pth' ,map_location= args.device)
        online_data = torch.load('/home/zkx/ImgCap/CaDReL/captioning/Camel/saved/129.72.pth' ,map_location= args.device)
        target_model.load_state_dict(target_data['state_dict_t'])
        online_model.load_state_dict(online_data['state_dict_t'])
        target_model.to(args.device)
        online_model.to(args.device)

    # test(args=args)
    # test(args, image_model, target_model, online_model, datasets, image_field, text_field)
    # train(args, image_model, target_model, online_model, datasets, image_field, text_field)