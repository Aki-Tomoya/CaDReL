import torch
import argparse
from common.train import train
from common.data.field import ImageDetectionsField, TextField
from common.utils.utils import create_dataset
from .models import build_encoder, build_decoder
from common.models.transformer.transformer import Transformer


def parse_args():
    parser = argparse.ArgumentParser(description='Memory Transformer')
    parser.add_argument('--output', type=str, default='AttModel')
    parser.add_argument('--exp_name', type=str, default='butd')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--head', type=int, default=8)

    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)

    parser.add_argument('--xe_base_lr', type=float, default=1e-4)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--use_rl', action='store_true')
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--feature_type', type=str, default='butd')
    # parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_FR_REGION.hdf5')
    parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_ViT-B-32_GLOBAL.hdf5')
    # parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_swin_base_patch4_window7_224_in22k.hdf5')
    # parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_RN50x4_GLOBAL.hdf5')
    # parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_VinVL.hdf5')
    parser.add_argument('--image_folder', type=str, default='coco/images')
    parser.add_argument('--annotation_folder', type=str, default='coco/annotations')
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    print('Memory Transformer Training')

    # Pipeline for image regions
    # image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50)
    image_field = ImageDetectionsField(feature_type=args.feature_type, detections_path=args.features_path, max_detections=50, global_feature=False)
    d_in = 2048

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    datasets = create_dataset(args, image_field, text_field)

    encoder = build_encoder(args.N_enc, 0, d_in=d_in)
    decoder = build_decoder(len(text_field.vocab), 54, args.N_dec, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(args.device)

    train(args, model, datasets, image_field, text_field)