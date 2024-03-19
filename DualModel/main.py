import argparse
# from DualModel.utils.train import train
from common.data.field import DualImageField, DualImageWithSemanticField, DualImageWithTextField, ImageDetectionsField, TextField, TextWithLabelField
# from common.models.transformer.transformer import Transformer
from DualModel.models.transformer import Transformer
from common.train import train
import clip
from common.utils.utils import create_dataset
from .utils.utils import *
from .models import build_encoder, build_decoder


def parse_args():
    parser = argparse.ArgumentParser(description='Dual Transformer')
    parser.add_argument('--output', type=str, default='DualModel')
    parser.add_argument('--exp_name', type=str, default='dual_val2')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--labels_length', type=int, default=10)

    parser.add_argument('--xe_base_lr', type=float, default=1e-4)
    parser.add_argument('--rl_base_lr', type=float, default=5e-6)
    parser.add_argument('--use_rl', action='store_true')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--with_pe', type=str, default=None, choices=['rpe', 'ape', 'dpe'])
    parser.add_argument('--feature_type', type=str, default='both')
    parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_RN50x4_GLOBAL.hdf5')
    parser.add_argument('--image_folder', type=str, default='coco/images')
    parser.add_argument('--annotation_folder', type=str, default='coco/annotations')
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    print('Dual Transformer Training')

    # Pipeline for image features
    # image_field = ImageDetectionsField(feature_type='clip', detections_path='coco/features/COCO2014_RN50x4.hdf5')
    image_field = DualImageField(max_detections=50, global_feature=False)
    # image_field = DualImageWithSemanticField(k=12, max_detections=50)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    # text_field = TextWithLabelField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
    #                        remove_punctuation=True, nopoints=False, labels_length=args.labels_length)

    # IT_field = DualImageWithTextField(labels_length=args.labels_length, max_detections=50, lower=True, tokenize='spacy',
    #                        remove_punctuation=True, nopoints=False)
                           
    datasets = create_dataset(args, image_field, text_field)

    encoder = build_encoder(args.N_enc, prefix_length=args.prefix_length, with_pe=args.with_pe, device=args.device)
    decoder = build_decoder(len(text_field.vocab), 54, args.N_dec, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(args.device)

    # clip_model, _ = clip.load('RN50x4', device=args.device, jit=False)
    # clip_model = clip_model.to(args.device).eval()

    train(args, model, datasets, image_field, text_field)
    # train(args, model, datasets, image_field, text_field, train_xe_fn=train_sm_xe, evaluate_loss_fn=evaluate_sm_loss)
    # train(args, clip_model, model, datasets, image_field, text_field, train_xe_fn=train_xe, evaluate_loss_fn=evaluate_loss)