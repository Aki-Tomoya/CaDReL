# coding: utf8
import base64
import unicodedata
from collections import Counter, OrderedDict
import json
import pickle
import random
from torch.nn import functional as F
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.folder import default_loader
from multiprocessing.pool import ThreadPool as Pool
from .tokenizer.simple_tokenizer import SimpleTokenizer as _Tokenizer
from itertools import chain
from PIL import Image
from itertools import takewhile
from tqdm import tqdm
import six
import torch
import numpy as np
import h5py
import os
import warnings
import shutil

from common.utils.tsv_file import TSVFile
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor,InterpolationMode
from .vocab import Vocab
from .utils import get_tokenizer


class RawField(object):
    """ Defines a general datatype.

    Every dataset consists of one or more types of data. For instance,
    a machine translation dataset contains paired examples of text, while
    an image captioning dataset contains images and texts.
    Each of these types of data is represented by a RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class Merge(RawField):
    def __init__(self, *fields):
        super(Merge, self).__init__()
        self.fields = fields

    def preprocess(self, x):
        return tuple(f.preprocess(x) for f in self.fields)

    def process(self, batch, *args, **kwargs):
        if len(self.fields) == 1:
            batch = [batch, ]
        else:
            batch = list(zip(*batch))

        out = list(f.process(b, *args, **kwargs) for f, b in zip(self.fields, batch))
        return out


class ImageDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, feature_type='butd', detections_path=None, max_detections=100,
                 with_pe=False, sort_by_prob=False, load_in_tmp=False, global_feature=False):
        self.max_detections = max_detections
        self.detections_path = detections_path
        self.feature_type = feature_type
        self.sort_by_prob = sort_by_prob
        self.with_pe = with_pe
        self.global_feature = global_feature

        tmp_detections_path = os.path.join('/tmp', os.path.basename(detections_path))

        if load_in_tmp:
            if not os.path.isfile(tmp_detections_path):
                if shutil.disk_usage("/tmp")[-1] < os.path.getsize(detections_path):
                    warnings.warn('Loading from %s, because /tmp has no enough space.' % detections_path)
                else:
                    warnings.warn("Copying detection file to /tmp")
                    shutil.copyfile(detections_path, tmp_detections_path)
                    warnings.warn("Done.")
                    self.detections_path = tmp_detections_path
            else:
                self.detections_path = tmp_detections_path

        available_features = ['butd', 'clip', 'vinvl', 'tokens']
        assert self.feature_type in available_features, \
               "region feature not supported, please select ['butd', 'clip', 'vinvl', 'tokens']"

        if self.feature_type in ['butd', 'vinvl', 'clip', 'tokens']:
            self.f = h5py.File(self.detections_path, 'r')

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id, split, orig_size = x['image_id'], x['split'], x['orig_size']
        try:
            if self.feature_type in ['butd', 'vinvl']:
                precomp_data = torch.from_numpy(self.f['%d_features' % image_id][()])
                if self.with_pe:
                    boxes = torch.from_numpy(self.f['%d_boxes' % image_id][()])
                    if len(boxes):
                        precomp_data = precomp_data[:len(boxes),:]

                if self.sort_by_prob:
                    idxs = torch.from_numpy(np.argsort(np.max(self.f['%d_cls_prob' % image_id][()], -1))[::-1])
                    precomp_data = precomp_data[idxs]
                    if self.with_pe:
                        boxes = boxes[idxs]

            # elif self.feature_type == 'vinvl':
            #     f, label, key2idx = self.fmap[split]
            #     feat_info = json.loads(f.seek(key2idx[str(image_id)])[1])
            #     precomp_data = torch.tensor(np.frombuffer(base64.b64decode(feat_info['features']), np.float32
            #             ).reshape((feat_info['num_boxes'], -1)))[:,:-6]
            #     if self.with_pe:
            #         labels = json.loads(label.seek(key2idx[str(image_id)])[1])
            #         if len(labels) == 0:
            #             raise KeyError
            #         else:
            #             precomp_data = precomp_data[:len(labels),:]
            #             boxes = torch.stack([torch.tensor(l['rect']) for l in labels])

            elif self.feature_type == 'clip':
                precomp_data = torch.from_numpy(self.f['%d_features' % image_id][()])
                if self.global_feature:
                    global_feature = torch.from_numpy(self.f['%d_global' % image_id][()])
                    return precomp_data, global_feature
                return precomp_data
            
            elif self.feature_type == 'tokens':
                precomp_data = torch.from_numpy(self.f['%d_tokens' % image_id][()])
                return precomp_data

            if self.with_pe:
                size = torch.tensor(orig_size).repeat(len(boxes), 2)
                relative_boxes = boxes / size
                
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = torch.rand(10,2048)
            relative_boxes = torch.rand((10, 4))

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = torch.cat([precomp_data, torch.zeros((delta, precomp_data.shape[1]))], 0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        if self.with_pe:
            delta_boxes = self.max_detections - len(relative_boxes)
            if delta_boxes > 0:
                relative_boxes = torch.cat([relative_boxes, torch.zeros((delta_boxes, relative_boxes.shape[1]))], 0)
            elif delta_boxes < 0:
                relative_boxes = relative_boxes[:self.max_detections]

            return (precomp_data, relative_boxes)

        return precomp_data

def build_transform(is_train, config):
    from timm.data import create_transform
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from timm.data.transforms import _pil_interp
    from torchvision import transforms

    resize_im = config.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.IMG_SIZE,
            is_training=True,
            color_jitter=config.COLOR_JITTER if config.COLOR_JITTER > 0 else None,
            auto_augment=config.AUTO_AUGMENT if config.AUTO_AUGMENT != 'none' else None,
            re_prob=config.REPROB,
            re_mode=config.REMODE,
            re_count=config.RECOUNT,
            interpolation=config.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TESTCROP:
            size = int((256 / 224) * config.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE),
                                interpolation=_pil_interp(config.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class ImageField(RawField):
    def __init__(self, config=None, preprocessing=None, postprocessing=None, loader=default_loader):
        self.loader = loader
        # self.transform = transform        
        self.train_transform = build_transform(True, config)
        self.test_transform = build_transform(False, config)
        self.txtCtxField = TxtCtxField('coco/features/txt_ctx.hdf5', 12)
        super().__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        # sample = self.loader(x['image_path'])
        image_path = x['image_path']
        image_id = x['image_id']
        is_train = x['split'] == 'train'

        img = Image.open(image_path).convert('RGB')
        if is_train:
            grid_feats = self.train_transform(img)
        else:
            grid_feats = self.test_transform(img)
        
        semantic_feats = self.txtCtxField.preprocess(image_id)
        return grid_feats, semantic_feats

    def process(self, batch):
        grid_feats, semantic_feats = zip(*batch)
        grid_features = default_collate(grid_feats)
        txt_ctx = self.txtCtxField.process(semantic_feats)
        return grid_features, txt_ctx

class ImageWithDAField(RawField):
    def __init__(self,train_transform, test_transform, preprocessing=None, postprocessing=None, loader=default_loader):
        self.loader = loader
        self.train_transform = train_transform
        self.test_transform = test_transform
        super().__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        split = x['split']
        sample = self.loader(x['image_path'])

        if split == 'train':
            prob = random.uniform(0,1)
            if prob < 0.5:
                sample = self.train_transform(sample)
            else:
                sample = self.test_transform(sample)

        else:
            sample = self.test_transform(sample)
        return sample


class DualImageField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, max_detections=100, global_feature=False,
                 with_pe=False, sort_by_prob=False, load_in_tmp=False, online_test=False):
        # self.clip_field = ImageField(transform, preprocessing, postprocessing, loader)
        # clip_path = 'coco/features/COCO2014_RN50x4_GLOBAL.hdf5'
        # vinvl_path = 'coco/features/COCO2014_VinVL.hdf5'
        clip_path = 'coco/features/COCO2014_FR_GRID.hdf5'
        vinvl_path = 'coco/features/COCO2014_FR_REGION.hdf5'

        if online_test:
            clip_path = 'coco/features/COCO2014_RN50x4_GLOBAL_TEST.hdf5'
            vinvl_path = 'coco/features/COCO2014_VinVL_TEST.hdf5'

        self.clip_field = ImageDetectionsField(preprocessing, postprocessing, 'clip', clip_path, global_feature=global_feature)
        self.vinvl_field = ImageDetectionsField(preprocessing, postprocessing, 'vinvl', vinvl_path, 
                                                max_detections, with_pe, sort_by_prob, load_in_tmp)
        self.global_feature = global_feature
        super().__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        region_features =  self.vinvl_field.preprocess(x)
        if self.global_feature:
            grid_features, global_feature = self.clip_field.preprocess(x)
            return (grid_features, region_features, global_feature)
        else:
            grid_features = self.clip_field.preprocess(x)
            return (grid_features, region_features)

class ClipSemField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, online_test=False):
        clip_path = 'coco/features/COCO2014_RN50x4_GLOBAL.hdf5'
        self.sem_dir = 'coco/images/semantic_seg'
        if online_test:
            clip_path = 'coco/features/COCO2014_RN50x4_GLOBAL_TEST.hdf5'
        self.clip_field = ImageDetectionsField(preprocessing, postprocessing, 'clip', clip_path)
        self.transform = Compose([
            # lambda x: F.interpolate(x.unsqueeze(0), size=288, mode="nearest"),
            Resize(288, interpolation=InterpolationMode.NEAREST),
            CenterCrop(288),
            # lambda x: F.interpolate(x.unsqueeze(0), size=(9,9), mode="nearest"),
            Resize((9,9), interpolation=InterpolationMode.NEAREST),
            # ToTensor(),
            lambda x: torch.from_numpy(np.asarray(x)).long()
            ])
        super().__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        grid_features = self.clip_field.preprocess(x)
        image_path = "%d.png" % x['image_id']
        sem_img = Image.open(os.path.join(self.sem_dir, image_path))
        sem_gt = self.transform(sem_img)
        return (grid_features, sem_gt)

class DualImageWithTextField(RawField):
    def __init__(self, labels_length=10, preprocessing=None, postprocessing=None, max_detections=100,
                 with_pe=False, sort_by_prob=False, load_in_tmp=False, lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False):
        # self.clip_field = ImageField(transform, preprocessing, postprocessing, loader)
        self.clip_field = ImageDetectionsField(preprocessing, postprocessing, 'clip', 'coco/features/COCO2014_RN50x4_GLOBAL.hdf5')
        self.vinvl_field = ImageDetectionsField(preprocessing, postprocessing, 'vinvl', 'coco/features/COCO2014_VinVL.hdf5', 
                                                max_detections, with_pe, sort_by_prob, load_in_tmp)
        self.sem_field = TextField(lower=lower, tokenize=tokenize, remove_punctuation=remove_punctuation, 
                                    nopoints=nopoints, fix_length=labels_length)
        self.sem_field.vocab = pickle.load(open('cache/vocab.pkl', 'rb'))
        with open('coco/features/COCO2014_VinVL_labels.json') as f:
                self.semantic_table = json.load(f)
        super().__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        image_id = x['image_id']
        semantic_words = self.semantic_table[str(image_id)]
        grid_features =  self.clip_field.preprocess(x)
        region_features =  self.vinvl_field.preprocess(x)
        semantic_words = self.sem_field.preprocess({'caption': semantic_words})
        return grid_features, region_features, semantic_words

    def process(self, batch):
        grid_features, region_features, semantic_words = zip(*batch)
        grid_features = default_collate(grid_features)
        region_features = default_collate(region_features)
        semantic_gt = self.sem_field.process(semantic_words)
        return grid_features, region_features, semantic_gt

class DualImageWithSemanticField(RawField):
    def __init__(self, k=4, preprocessing=None, postprocessing=None, max_detections=100,
                 with_pe=False, sort_by_prob=False, load_in_tmp=False):
        # self.clip_field = ImageField(transform, preprocessing, postprocessing, loader)
        self.clip_field = ImageDetectionsField(preprocessing, postprocessing, 'clip', 'coco/features/COCO2014_RN50x4_GLOBAL.hdf5')
        self.vinvl_field = ImageDetectionsField(preprocessing, postprocessing, 'vinvl', 'coco/features/COCO2014_VinVL.hdf5', 
                                                max_detections, with_pe, sort_by_prob, load_in_tmp)
        self.txtCtxField = TxtCtxField('coco/features/txt_ctx.hdf5', k)
        super().__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        image_id = x['image_id']
        grid_features =  self.clip_field.preprocess(x)
        region_features =  self.vinvl_field.preprocess(x)
        semantic_features = self.txtCtxField.preprocess(image_id)
        return grid_features, region_features, semantic_features

    def process(self, batch):
        grid_features, region_features, semantic_features = zip(*batch)
        grid_features = default_collate(grid_features)
        region_features = default_collate(region_features)
        txt_ctx = self.txtCtxField.process(semantic_features)
        return grid_features, region_features, txt_ctx

class TxtCtxField(RawField):
    def __init__(self, ctx_file, k=4, preload=False, preprocessing=None, postprocessing=None):
        self.k = k

        self.preload = preload
        if preload:
            self.ctx = self.load(ctx_file)
        else:
            self.ctx = h5py.File(ctx_file, "r")

        super(TxtCtxField, self).__init__(preprocessing, postprocessing)
    
    def __load(self, ctx, k):
        ctx_whole_f = ctx[f"{k}/whole/features"][:self.k]

        ctx_five_f = ctx[f"{k}/five/features"][:, :self.k]
        ctx_five_p = np.tile(np.arange(5)[:, None], (1, self.k))
        ctx_five_f = ctx_five_f.reshape((5*self.k, -1))
        ctx_five_p = ctx_five_p.reshape((5*self.k, ))

        ctx_nine_f = ctx[f"{k}/nine/features"][:, :self.k]
        ctx_nine_p = np.tile(np.arange(9)[:, None], (1, self.k))
        ctx_nine_f = ctx_nine_f.reshape((9*self.k, -1))
        ctx_nine_p = ctx_nine_p.reshape((9*self.k, ))

        return {
            "whole": {"features": ctx_whole_f},
            "five": {"features": ctx_five_f, "positions": ctx_five_p},
            "nine": {"features": ctx_nine_f, "positions": ctx_nine_p}
        }
    
    def load(self, ctx_file):
        print(f"Preload features from {str(ctx_file)}...")
        ctx_file = h5py.File(ctx_file, "r")
        pool = Pool(128)
        results = {
            k: pool.apply_async(self.__load, args=(ctx_file, k))
            for k in ctx_file.keys()
        }
        ctx = {k: v.get() for k, v in tqdm(results.items())}
        ctx_file.close()

        return ctx
    
    def preprocess(self, x):
        if self.preload:
            ctx_whole_f = self.ctx[str(x)]["whole"]["features"]
            ctx_five_f = self.ctx[str(x)]["five"]["features"]
            ctx_five_p = self.ctx[str(x)]["five"]["positions"]
            ctx_nine_f = self.ctx[str(x)]["nine"]["features"]
            ctx_nine_p = self.ctx[str(x)]["nine"]["positions"]
        else:
            data = self.__load(self.ctx, x)
            ctx_whole_f = data["whole"]["features"]
            ctx_five_f = data["five"]["features"]
            ctx_five_p = data["five"]["positions"]
            ctx_nine_f = data["nine"]["features"]
            ctx_nine_p = data["nine"]["positions"]

        ctx_whole_f = torch.FloatTensor(ctx_whole_f)
        ctx_five_f = torch.FloatTensor(ctx_five_f)
        ctx_five_p = torch.LongTensor(ctx_five_p)
        ctx_nine_f = torch.FloatTensor(ctx_nine_f)
        ctx_nine_p = torch.LongTensor(ctx_nine_p)

        return ctx_whole_f, ctx_five_f, ctx_five_p, ctx_nine_f, ctx_nine_p
            
    def process(self, batch, *args, **kwargs):
        ctx_whole_f, ctx_five_f, ctx_five_p, ctx_nine_f, ctx_nine_p = list(zip(*batch))
        
        ctx_whole_f = torch.stack(ctx_whole_f)
        ctx_five_f = torch.stack(ctx_five_f)
        ctx_nine_f = torch.stack(ctx_nine_f)

        ctx_whole_p = torch.zeros((len(ctx_whole_f), len(ctx_whole_f[0])), dtype=torch.long)
        ctx_five_p = torch.stack(ctx_five_p)
        ctx_nine_p = torch.stack(ctx_nine_p)

        return {
            "whole": {"embed": ctx_whole_f, "pos": ctx_whole_p},
            "five": {"embed": ctx_five_f, "pos": ctx_five_p},
            "nine": {"embed": ctx_nine_f, "pos": ctx_nine_p},
        }


# class TextBPEField(RawField):
#     def __init__(self):
#         self._tokenizer = _Tokenizer()
#         super(TextField, self).__init__()

#     def preprocess(self, x):
#         if x is None:
#             return ''
#         return x

#     def process(self, texts):
#         if isinstance(texts, str):
#             texts = [texts]

#         sot_token = self._tokenizer.bos_idx
#         eot_token = self._tokenizer.eos_idx
#         all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
#         result = torch.zeros(len(all_tokens), max(len(s) for s in all_tokens), dtype=torch.long)

#         for i, tokens in enumerate(all_tokens):
#             result[i, :len(tokens)] = torch.tensor(tokens)

#         return result

#     def decode(self, word_idxs):
#         if isinstance(word_idxs, list) and len(word_idxs) == 0:
#             return self.decode([word_idxs, ])[0]
#         if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
#             return self.decode([word_idxs, ])[0]
#         elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
#             return self.decode(word_idxs.unsqueeze(0))[0]

#         captions = []
#         for wis in word_idxs:
#             wis = wis.tolist()
#             wis = list(takewhile(lambda tok: tok != self._tokenizer.eos_idx, wis))
#             caption = self._tokenizer.decode(wis)
#             captions.append(caption)
#         return captions

class TextField(RawField):
    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 remove_punctuation=False, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False, vectors=None, nopoints=True, reverse=False):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = None
        self.vectors = vectors
        self.reverse = reverse
        if nopoints:
            self.punctuations.append("..")

        super(TextField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        x = x['caption']
        if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.lower:
            x = six.text_type.lower(x)
        x = self.tokenize(x.rstrip('\n'))
        if self.remove_punctuation:
            x = [w for w in x if w not in self.punctuations]
        if self.preprocessing is not None:
            x =  self.preprocessing(x)

        if self.reverse:
            return x, list(reversed(x))
        else:
            return x

    def process(self, batch, device=None):
        if self.reverse:
            batch = list(zip(*batch))
            padded_1 = self.pad(batch[0])
            padded_2 = self.pad(batch[1], reverse=True)
            tensor_1 = self.numericalize(padded_1, device=device)
            tensor_2 = self.numericalize(padded_2, device=device)
            return tensor_1, tensor_2
            # padded = self.pad(batch, reverse=True)
            # tensor = self.numericalize(padded, device=device)
            # return tensor
        else:
            padded = self.pad(batch)
            tensor = self.numericalize(padded, device=device)
            return tensor

    def build_vocab(self, *args, **kwargs):
        from .dataset import Dataset
        
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials = list(OrderedDict.fromkeys([
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None]))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def pad(self, minibatch, reverse=False):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            elif reverse:
                padded.append(
                    ([] if self.eos_token is None else [self.eos_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.init_token is None else [self.init_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a list of Variables.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)

            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            arr = [numericalization_func(x) if isinstance(x, six.string_types)
                   else x for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.eos_token:
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions


class TextBPEField(RawField):
    def __init__(self):
        self._tokenizer = _Tokenizer()
        super(TextField, self).__init__()

    def preprocess(self, x):
        if x is None:
            return ''
        return x

    def process(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self._tokenizer.bos_idx
        eot_token = self._tokenizer.eos_idx
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), max(len(s) for s in all_tokens), dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def decode(self, word_idxs):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ])[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ])[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0))[0]

        captions = []
        for wis in word_idxs:
            wis = wis.tolist()
            wis = list(takewhile(lambda tok: tok != self._tokenizer.eos_idx, wis))
            caption = self._tokenizer.decode(wis)
            captions.append(caption)
        return captions

class TextWithSemanticField(RawField):
        def __init__(self, init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False):
            with open('cache/semantic_words.json') as f:
                self.semantic_table = json.load(f)

            self.text_field = TextField(init_token=init_token, eos_token=eos_token, lower=lower, tokenize=tokenize, 
                                    remove_punctuation=remove_punctuation, nopoints=nopoints)
            self.semantic_vocab = None

        def preprocess(self, x):
            ann_id, caption = x['ann_id'], x['caption']
            semantic_words = self.semantic_table[str(ann_id)]

            caption = self.text_field.preprocess(caption)
            semantic_words = self.text_field.preprocess(semantic_words)

            return (caption, semantic_words)

        def process(self, batch, device=None):
            caption, semantic_words = zip(*batch)
            caption_gt = self.text_field.process(caption, device)
            b_s = len(batch)
            semantic_words = [[self.semantic_vocab.stoi[x] for x in ex] for ex in semantic_words]
            # semantic_words = torch.tensor(semantic_words, dtype=torch.long, device=device)
            semantic_gt = torch.zeros((b_s, len(self.semantic_vocab)), dtype=torch.long , device=device)
            # semantic_gt.scatter_(1, semantic_words, 1)
            for i in range(b_s):
                for j in semantic_words[i]:
                    semantic_gt[i, j] = 1

            return caption_gt, semantic_gt

        def build_vocab(self, *args, **kwargs):
            from .dataset import Dataset
            
            semantic_counter = Counter()
            sources = []
            for arg in args:
                if isinstance(arg, Dataset):
                    sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
                else:
                    sources.append(arg)

            for data in sources:
                for x in data:
                    caption, semantic_words = self.preprocess(x)
                    try:
                        semantic_counter.update(semantic_words)
                    except TypeError:
                        semantic_counter.update(chain.from_iterable(semantic_counter))

            self.semantic_vocab = self.vocab_cls(semantic_counter, specials=[], **kwargs)


class TextWithLabelField(RawField):
        def __init__(self, init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False, labels_length=10):
            # with open('coco/features/COCO2014_VinVL_labels.json') as f:
            with open('coco/features/COCO2014_VinVL_labels.json') as f:
                self.semantic_table = json.load(f)
            vocab_path = 'cache/vocab.pkl'
            self.vocab = pickle.load(open(vocab_path, 'rb'))
            self.cap_field = TextField(init_token=init_token, eos_token=eos_token, lower=lower, tokenize=tokenize, 
                                    remove_punctuation=remove_punctuation, nopoints=nopoints)
            self.sem_field = TextField(lower=lower, tokenize=tokenize, remove_punctuation=remove_punctuation, 
                                     nopoints=nopoints, fix_length=labels_length)
            self.cap_field.vocab = self.vocab
            self.sem_field.vocab = self.vocab
        def preprocess(self, x):
            image_id, caption = x['image_id'], x['caption']
            semantic_words = self.semantic_table[str(image_id)]

            caption = self.cap_field.preprocess({'caption': caption})
            semantic_words = self.sem_field.preprocess({'caption': semantic_words})
            return (caption, semantic_words)

        def process(self, batch, device=None):
            caption, semantic_words = zip(*batch)
            caption_gt = self.cap_field.process(caption, device)
            semantic_gt = self.sem_field.process(semantic_words, device)

            return caption_gt, semantic_gt

        def decode(self, word_idxs, join_words=True):
            return self.cap_field.decode(word_idxs, join_words)

class DetectionsField(RawField):
    def __init__(self, transform, preprocessing=None, postprocessing=None):
        super(DetectionsField, self).__init__(preprocessing, postprocessing)
        self.transform = transform

    def preprocess(self, target):
        w, h = target['orig_size']

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        
        target["image_id"] = image_id
        target["orig_size"] = torch.tensor([int(h), int(w)])
        target["size"] = (w, h)

        return self.transform(target)

    def process(self, batch):
        return list(batch)

class ArtEmisGridField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, features_path=None, global_feature=False):
        self.global_feature = global_feature
        self.f = h5py.File(features_path, 'r')

        super(ArtEmisGridField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        x = unicodedata.normalize('NFC', x)
        precomp_data = torch.from_numpy(self.f['%s_features' % x][()])
        if self.global_feature:
            global_feature = torch.from_numpy(self.f['%d_global' % x][()])
            return precomp_data, global_feature
        return precomp_data

class EmotionField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, emotions=None):
        if emotions is None:
            emotions = [
                'amusement', 'awe', 'contentment', 'excitement', 
                'anger', 'disgust', 'fear', 'sadness', 'something else'
                ]
        self.emotion_mapping = { key: value for value, key in enumerate(emotions)}
        super(EmotionField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        return self.emotion_mapping[x]