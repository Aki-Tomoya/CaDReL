import torch
import h5py
import os, pickle
from Camel.models.captioner import Captioner
from common.data.field import TextField
from common.models import Transformer

def visualize(model, grid_features, text_field, device):
    model.eval()
    with torch.no_grad():
        grid_features = grid_features.unsqueeze(0).to(device)
        out, _ = model.beam_search(grid_features, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
    
    caps_gen = text_field.decode(out, join_words=False)
    caps_gen = ' '.join(caps_gen[0]).strip()
    return caps_gen

def get_features_by_id(image_id, max_detections=50):
    clip_path = '/home/public/caption/caption/coco2014/feats/COCO2014_RN50x4_GLOBAL.hdf5'
    # vinvl_path = 'coco/features/COCO2014_VinVL.hdf5'
    clip_file = h5py.File(clip_path, 'r')
    # vinvl_file = h5py.File(vinvl_path, 'r')

    feature_key = '%d_features' % image_id
    boxs_key = '%d_boxes' % image_id
    gird_feature = torch.from_numpy(clip_file[feature_key][()])
    # region_feature = torch.from_numpy(vinvl_file[feature_key][()])
    # boxes = torch.from_numpy(vinvl_file[boxs_key][()])

    # delta = max_detections - region_feature.shape[0]
    # if delta > 0:
    #     region_feature = torch.cat([region_feature, torch.zeros((delta, region_feature.shape[1]))], 0)
    # elif delta < 0:
    #     region_feature = region_feature[:max_detections]

    # return gird_feature, region_feature, boxes
    return gird_feature

def test(args=None):
    image_id = 2434
    device = args.device
    # model_path = 'coco/checkpoints/DualModel/dual_add_best.pth'
    # model_path = 'coco/checkpoints/DualModel/dual_fuse_global_best.pth'
    # model_path = '/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/141.11.pth'

    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                            remove_punctuation=True, nopoints=False)
    vocab_path = '/home/zkx/ImgCap/Discrimitor/captioning/cache/vocab.pkl'
    text_field.vocab = pickle.load(open(vocab_path, 'rb'))

    online_model = Captioner(args, text_field, istarget=1).to(args.device)
    # online_data = torch.load('/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/141.11.pth',map_location= args.device)
    online_data = torch.load('/home/zkx/ImgCap/Discrimitor/captioning/Camel/saved/base_119.pth',map_location= args.device)
    # online_model.load_state_dict(online_data['state_dict_t'])
    online_model.load_state_dict(online_data['state_dict_o'])
    online_model.to(args.device)
    


    print(online_data['best_cider'])

    # gird_feature, region_feature, boxes =  get_features_by_id(image_id)
    grid_feature =  get_features_by_id(image_id)
    # dual_features = [gird_feature.unsqueeze(0), region_feature.unsqueeze(0) ]
    # caps_gen = visualize(model, dual_features, text_field, device)
    caps_gen = visualize(online_model, grid_feature, text_field, device)
    print(caps_gen)

