import os
import numpy as np
import itertools
import collections
import torch
import pandas as pd

from .example import Example
from .utils import nostdout
from pycocotools.coco import COCO as pyCOCO

class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def collate_fn(self):
        def collate(batch):
            if len(self.fields) == 1:
                batch = [batch, ]
            else:
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                # if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                #     tensors.extend(tensor)
                # else:
                tensors.append(tensor)

            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]

        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        for field_name, field in self.fields.items():
            data.append(field.preprocess(getattr(example, field_name)))

        if len(data) == 1:
            data = data[0]
        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(Dataset):
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return value_tensors
        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(Dataset):
    def __init__(self, examples, fields, key_fields, val_fields=None):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        if (val_fields is not None) and (not isinstance(val_fields, (tuple, list))):
            val_fields = (val_fields,)
        for field in key_fields:
            assert (field in fields)

        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        if val_fields is not None:
            value_fields = {k: fields[k] for k in val_fields}
        else:
            value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            return key_tensors, value_tensors
        return collate

    def __getitem__(self, i):
        # TODO
        captions = []
        for item in self.value_dataset[i]:
            captions.append(item['caption'])
        return self.key_dataset[i], captions
        # return self.key_dataset[i], self.value_dataset[i]

        # arr = []
        # for item in self.value_dataset[i]:
        #     arr.append([item[0]['caption'], item[1]])
        # return self.key_dataset[i], arr

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        assert ('image' in fields)
        assert ('text' in fields)
        super(PairedDataset, self).__init__(examples, fields)
        self.image_field = self.fields['image']
        self.text_field = self.fields['text']

    def image_set(self):
        img_list = [e.image for e in self.examples]
        image_set = unique(img_list)
        examples = [Example.fromdict({'image': i}) for i in image_set]
        dataset = Dataset(examples, {'image': self.image_field})
        return dataset

    def text_set(self):
        text_list = [e.text for e in self.examples]
        text_list = unique(text_list)
        examples = [Example.fromdict({'text': t}) for t in text_list]
        dataset = Dataset(examples, {'text': self.text_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='image')
        return dataset

    def text_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='text')
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class TripleDataset(Dataset):
    def __init__(self, examples, fields):
        assert ('image' in fields)
        assert ('text' in fields)
        assert('emotion' in fields)
        super(TripleDataset, self).__init__(examples, fields)
        self.image_field = self.fields['image']
        self.text_field = self.fields['text']
        self.emotion_field = self.fields['emotion']

    def image_set(self):
        img_list = [e.image for e in self.examples]
        image_set = unique(img_list)
        examples = [Example.fromdict({'image': i}) for i in image_set]
        dataset = Dataset(examples, {'image': self.image_field})
        return dataset

    def text_set(self):
        text_list = [e.text for e in self.examples]
        text_list = unique(text_list)
        examples = [Example.fromdict({'text': t}) for t in text_list]
        dataset = Dataset(examples, {'text': self.text_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='image')
        return dataset

    def text_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='text')
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class COCODataset(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root, id_root=None, use_restval=True,
                 cut_validation=False):
        roots = {}
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }

        if id_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(id_root, 'coco_train_ids.npy'))
            ids['val'] = np.load(os.path.join(id_root, 'coco_dev_ids.npy'))
            if cut_validation:
                ids['val'] = ids['val'][:5000]
            ids['test'] = np.load(os.path.join(id_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (
                ids['train'],
                np.load(os.path.join(id_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None

        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCODataset, self).__init__(examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, roots, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ['train', 'val', 'test']:
            if isinstance(roots[split]['cap'], tuple):
                coco_dataset = (pyCOCO(roots[split]['cap'][0]), pyCOCO(roots[split]['cap'][1]))
                root = roots[split]['img']
            else:
                coco_dataset = (pyCOCO(roots[split]['cap']),)
                root = (roots[split]['img'],)

            if ids_dataset is None:
                ids = list(coco_dataset.anns.keys())
            else:
                ids = ids_dataset[split]

            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])
            else:
                bp = len(ids)

            for index in range(len(ids)):
            # dataset change
            # for index in range(40):
                if index < bp:
                    coco = coco_dataset[0]
                    img_root = root[0]
                else:
                    coco = coco_dataset[1]
                    img_root = root[1]

                ann_id = ids[index]
                ann = coco.anns[ann_id]
                caption = ann['caption']
                img_id = ann['image_id']
                image = coco.imgs[img_id]

                image_path = os.path.join(img_root, image['file_name'])
                orig_size = (image['width'], image['height'])

                image_des = {'image_id': img_id, 'image_path': image_path, 'split': split, 'orig_size': orig_size}
                text = {'image_id': img_id, 'ann_id': ann_id, 'caption': caption}

                example = Example.fromdict({'image': image_des, 'text': text})

                if split == 'train':
                    train_samples.append(example)
                elif split == 'val':
                    val_samples.append(example)
                elif split == 'test':
                    test_samples.append(example)

        return train_samples, val_samples, test_samples


class OnlineTestDataset(Dataset):
    def __init__(self, ann_path, fields):
        examples = self.get_samples(ann_path)
        # examples = examples[:20]
        super(OnlineTestDataset, self).__init__(examples, fields)

    def get_samples(self, ann_path):
        coco_dataset = pyCOCO(ann_path)
        test_samples = []
        for img_id, image in coco_dataset.imgs.items():
            orig_size = (image['width'], image['height'])
            image_des = {'image_id': img_id, 'split': 'test', 'orig_size': orig_size}
            example = Example.fromdict({'image': image_des, 'image_id': int(img_id)})
            test_samples.append(example)
        return test_samples


class COCOWithDetectionDataset(Dataset):
    def __init__(self, fields, img_root, ann_root, id_root=None, use_restval=True,
                 cut_validation=False):
        self.fields = fields
        roots = {}
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json'),
            'det': os.path.join(ann_root, 'instances_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json'),
            'det': os.path.join(ann_root, 'instances_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json'),
            'det': os.path.join(ann_root, 'instances_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap']),
            'det': (roots['train']['det'], roots['val']['det'])
        }

        if id_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(id_root, 'coco_train_ids.npy'))
            ids['val'] = np.load(os.path.join(id_root, 'coco_dev_ids.npy'))
            if cut_validation:
                ids['val'] = ids['val'][:5000]
            ids['test'] = np.load(os.path.join(id_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (
                ids['train'],
                np.load(os.path.join(id_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None

        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCOWithDetectionDataset, self).__init__(examples, fields)

    @property
    def splits(self):
        from common.data.field import RawField

        train_split = Dataset(self.train_examples, self.fields)
        val_split = Dataset(self.val_examples, self.fields)
        test_split = Dataset(self.test_examples, self.fields)
        fields = {'image': self.fields['image'], 'text': RawField()}
        train_DDset = DictionaryDataset(self.train_examples, fields, key_fields='image')
        val_DDset = DictionaryDataset(self.val_examples, fields, key_fields='image')
        test_DDset = DictionaryDataset(self.test_examples, fields, key_fields='image')
        return (train_split, val_split, test_split, train_DDset, val_DDset, test_DDset)

    @classmethod
    def get_samples(cls, roots, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ['train', 'val', 'test']:
            if isinstance(roots[split]['cap'], tuple):
                cap_dataset = (pyCOCO(roots[split]['cap'][0]), pyCOCO(roots[split]['cap'][1]))
                det_dataset = (pyCOCO(roots[split]['det'][0]), pyCOCO(roots[split]['det'][1]))
                root = roots[split]['img']
            else:
                cap_dataset = (pyCOCO(roots[split]['cap']),)
                root = (roots[split]['img'],)

            if ids_dataset is None:
                ids = list(cap_dataset.anns.keys())
            else:
                ids = ids_dataset[split]

            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])
            else:
                bp = len(ids)

            for index in range(len(ids)):
            # for index in range(5000):
                if index < bp:
                    cap_coco = cap_dataset[0]
                    det_coco = det_dataset[0]
                    img_root = root[0]
                else:
                    cap_coco = cap_dataset[1]
                    det_coco = det_dataset[0]
                    img_root = root[1]

                ann_id = ids[index]
                img_id = cap_coco.anns[ann_id]['image_id']
                caption = cap_coco.anns[ann_id]['caption']
                det_ids = det_coco.getAnnIds(img_id)
                target = det_coco.loadAnns(det_ids)

                image = cap_coco.imgs[img_id]
                filename = os.path.join(img_root, image['file_name'])
                detection = {'orig_size': (image['width'], image['height']), 
                            'image_id':img_id, 'annotations': target}

                example = Example.fromdict({'image': filename, 'text': caption, 'detection': detection})

                if split == 'train':
                    train_samples.append(example)
                elif split == 'val':
                    val_samples.append(example)
                elif split == 'test':
                    test_samples.append(example)

        return train_samples, val_samples, test_samples


class ArtEmis(TripleDataset):
    def __init__(self, image_field, text_field, emotion_field, ann_root):
        self.df = pd.read_csv(ann_root)
        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(self.df)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(ArtEmis, self).__init__(examples, {'image': image_field, 'text': text_field, 'emotion': emotion_field})

    @property
    def splits(self):
        train_split = TripleDataset(self.train_examples, self.fields)
        val_split = TripleDataset(self.val_examples, self.fields)
        test_split = TripleDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, df):
        train_samples = []
        val_samples = []
        test_samples = []
        cnt = 0

        for i, row in df.iterrows():
            # if cnt == 100:
            #     break
            # cnt += 1
            split = row['split']
            painting = row['painting']
            style = row['art_style']
            caption = row['utterance']
            if split == 'test':
                emotion = row['grounding_emotion']
            else:
                emotion = row['emotion']
            filename = '/' + style + '/' + painting

            data_entry = {
                'image': filename, 
                'text': {'caption': caption},
                'emotion': emotion
                }

            example = Example.fromdict(data_entry)
            if split == 'train':
                train_samples.append(example)
            elif split == 'val':
                val_samples.append(example)
            elif split == 'test':
                test_samples.append(example)
        
        return train_samples, val_samples, test_samples