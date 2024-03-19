import torch
import argparse
from ClipModel.utils.train import train
from ClipModel.utils.utils import create_transform
from common.models.backbone import clip
from common.data.field import ImageDetectionsField, ImageWithDAField, TextField
from common.utils.utils import create_dataset
from .models import build_encoder, build_decoder
from common.models.transformer.transformer import Transformer


def parse_args():
    parser = argparse.ArgumentParser(description='ClipBase Transformer')
    parser.add_argument('--output', type=str, default='ClipModel')
    parser.add_argument('--exp_name', type=str, default='vanilla_val')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)

    parser.add_argument('--xe_base_lr', type=float, default=1e-4)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--use_rl', action='store_true')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--multi_level', action='store_true')
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--mid_dim', type=int, default=40)
    parser.add_argument('--with_pe', type=str, default=None, choices=['rpe', 'ape', 'dpe'])
    # parser.add_argument('--feature_type', type=str, default='tokens')
    # parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_VisualTokens.hdf5')
    parser.add_argument('--feature_type', type=str, default='clip')
    parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_RN50x4_GLOBAL.hdf5')
    parser.add_argument('--image_folder', type=str, default='coco/images')
    parser.add_argument('--annotation_folder', type=str, default='coco/annotations')
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    print('ClipBase Transformer Training')

    # Pipeline for image features
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
                           remove_punctuation=True, nopoints=False)

    datasets = create_dataset(args, image_field, text_field)

    encoder = build_encoder(args.N_enc, d_in=2560, with_pe=args.with_pe, mid_dim=args.mid_dim, multi_level=args.multi_level, local=args.local, device=args.device)
    decoder = build_decoder(len(text_field.vocab), 54, args.N_dec, text_field.vocab.stoi['<pad>'], multi_level=args.multi_level, aux_loss=args.aux_loss)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(args.device)

    train(args, image_model, model, datasets, image_field, text_field)