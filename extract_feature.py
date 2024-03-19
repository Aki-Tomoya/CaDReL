import h5py
import os
import torch
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os.path as osp
import unicodedata
from pycocotools.coco import COCO
from common.models.backbone import clip

def extract_feature(clip_variant, device):
    # image_dir = '/home/public/caption/caption/coco2014/images/test2014'
    # base_path = '/home/public/caption/caption/coco2014/annotations/image_info_test2014.json'

    # save_path = os.path.join('/home/public/caption/caption/coco2014/feats', 'COCO2014_%s_TEST.hdf5' % clip_variant)
    # # save_path = os.path.join('/home/public/caption/coco2014/feats', 'COCO2014_%s_GLOBAL.hdf5' % clip_variant)
    # f = h5py.File(save_path, mode='w')

    # clip_model, transform = clip.load(clip_variant, device=device ,jit=False)
    # image_model = clip_model.visual.to(device).eval()
    # image_model.forward = image_model.intermediate_features
    # coco = COCO(base_path)

    # with torch.no_grad():
    #     for img_id, img in tqdm(coco.imgs.items()):
    #         image_path = os.path.join(image_dir,img['file_name'])

    #         image = Image.open(image_path).convert('RGB')
    #         image = transform(image)

    #         image = image.to(device).unsqueeze(0)
    #         gird, x = image_model.forward2(image)

    #         gird = gird.squeeze(0).cpu().numpy()
    #         x = x.squeeze(0).cpu().numpy()
    #         f.create_dataset('%s_features' % img_id, data=gird)
    #         f.create_dataset('%s_global' % img_id, data=x)

    # f.close()

    # for split in ['test']:
    #     ann_path = base_path % split
    #     coco = COCO(ann_path)

    #     with torch.no_grad():
    #         for img_id, img in tqdm(coco.imgs.items(), split):
    #             image_path = os.path.join(image_dir % split ,img['file_name'])

    #             image = Image.open(image_path).convert('RGB')
    #             image = transform(image)

    #             image = image.to(device).unsqueeze(0)
    #             gird, x = image_model.forward2(image)

    #             gird = gird.squeeze(0).cpu().numpy()
    #             x = x.squeeze(0).cpu().numpy()
    #             f.create_dataset('%s_features' % img_id, data=gird)
    #             f.create_dataset('%s_global' % img_id, data=x)

    # f.close()

    image_dir = '/home/public/caption/caption/coco2014/images/test2014'
    save_path = os.path.join('/home/public/caption/caption/coco2014/feats', 'COCO2014_%s_GLOBAL_TEST.hdf5' % clip_variant)
    f = h5py.File(save_path, mode='w')

    clip_model, transform = clip.load(clip_variant, device=device ,jit=False)
    image_model = clip_model.visual.to(device).eval()
    image_model.forward = image_model.intermediate_features

    ann_path = '/home/public/caption/caption/coco2014/annotations/image_info_test2014.json'
    coco = COCO(ann_path)

    with torch.no_grad():
        for img_id, img in tqdm(coco.imgs.items()):
            image_path = os.path.join(image_dir ,img['file_name'])

            image = Image.open(image_path).convert('RGB')
            image = transform(image)

            image = image.to(device).unsqueeze(0)
            gird, x = image_model.forward2(image)

            gird = gird.squeeze(0).cpu().numpy()
            x = x.squeeze(0).cpu().numpy()
            f.create_dataset('%s_features' % img_id, data=gird)
            f.create_dataset('%s_global' % img_id, data=x)

    f.close()

def extract_artemis_feature(clip_variant, device):
    save_path = os.path.join('/home/public2/caption/coco2014/feats', 'ARTEMIS_%s_GLOBAL.hdf5' % clip_variant)
    f = h5py.File(save_path, mode='w')

    clip_model, transform = clip.load(clip_variant, device=device ,jit=False)
    image_model = clip_model.visual.to(device).eval()
    image_model.forward = image_model.intermediate_features

    df = pd.read_csv('/home/xys/code/artemis-v2/dataset/full_combined/train/artemis_preprocessed.csv')
    image_dir = '/home/xys/code/artemis-v2/dataset/wikiart'

    grouped_artwork = df.groupby(['art_style', 'painting'])
    print('Unique paintings annotated:', len(grouped_artwork.size()))

    with torch.no_grad():
        for group_name, _ in tqdm(grouped_artwork):
            art_style = group_name[0]
            painting = group_name[1]

            filename = '/' + art_style + '/' + painting
            filename = unicodedata.normalize('NFC', filename)

            image_path = osp.join(image_dir, art_style, painting + '.jpg')
            image_path = unicodedata.normalize('NFC', image_path)

            image = Image.open(image_path).convert('RGB')
            image = transform(image)

            image = image.to(device).unsqueeze(0)
            gird, x = image_model.forward2(image)

            gird = gird.squeeze(0).cpu().numpy()
            x = x.squeeze(0).cpu().numpy()
            f.create_dataset('%s_features' % filename, data=gird)
            f.create_dataset('%s_global' % filename, data=x)

    f.close()

if __name__=='__main__':
    clip_variant = 'RN50x16'
    device = 'cuda:0'
    extract_feature(clip_variant, device)