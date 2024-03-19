import random
import torch
import argparse
import numpy as np
from common.train import train, create_dataset
from common.data.field import ImageDetectionsField, TextField
from common.models.transformer.transformer import Transformer
from .models import build_encoder, build_decoder


def parse_args():
    parser = argparse.ArgumentParser(description='M2T')
    parser.add_argument('--output', type=str, default='M2T')
    parser.add_argument('--exp_name', type=str, default='M2T')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--use_rl', action='store_true')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str, default='coco_detections.hdf5')
    # parser.add_argument('--features_path', type=str, default='coco/vinvl')
    parser.add_argument('--image_folder', type=str, default='coco/images')
    parser.add_argument('--annotation_folder', type=str, default='coco/annotations')
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    print('Meshed-Memory Transformer Training')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
    # image_field = ImageDetectionsField(feature_name='vinvl', detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    datasets = create_dataset(args, image_field, text_field)

    encoder = build_encoder(3, 0, args.m)
    # decoder = build_decoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    decoder = build_decoder(len(text_field.vocab), args.m, 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(args.device)

    train(args, model, datasets, image_field, text_field)
    