import itertools
import multiprocessing
import os
import pickle
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop

from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

from common import evaluation
from common.data.dataset import COCODataset

def create_transform(n_px):
    train_transform = Compose([
        Resize(320, interpolation=BICUBIC),
        RandomCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    test_transform = Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    return train_transform, test_transform

def create_dataset(args, image_field, text_field):
    # Create the dataset
    dataset = COCODataset(image_field, text_field, args.image_folder, args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    vocab_path = 'cache/vocab.pkl'
    if not os.path.isfile(vocab_path):
        print("Building caption vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open(vocab_path, 'wb'))
    else:
        text_field.vocab = pickle.load(open(vocab_path, 'rb'))

    if args.enable_semantic:
        semantic_vocab_path = 'cache/semantic_vocab.pkl'
        if not os.path.isfile(semantic_vocab_path):
            print("Building semantic vocabulary")
            text_field.build_vocab(train_dataset, val_dataset, min_freq=10, max_size=2000)
            pickle.dump(text_field.semantic_vocab, open(semantic_vocab_path, 'wb'))
        else:
            text_field.semantic_vocab = pickle.load(open(semantic_vocab_path, 'rb'))

    return (train_dataset, val_dataset, test_dataset)

def train_xe(image_model, model, dataloader, optim, loss_fn, text_field, epoch, device = 'cuda', scheduler = None, args=None):
     # Training with cross-entropy
    model.train()
    if scheduler is not None:
        scheduler.step()
    # print('lr0 = ', optim.state_dict()['param_groups'][0]['lr'])
    running_loss = .0

    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions) in enumerate(dataloader):
            # if it == 10:
            #     break
            if isinstance(images, tuple) or isinstance(images, list):
                images = [x.to(device) for x in images]
            else:
                images = images.to(device)
            if isinstance(captions, tuple) or isinstance(captions, list):
                captions = [x.to(device) for x in captions]
            else:
                captions = captions.to(device)

            if args.online:
                with torch.no_grad():
                    images = image_model(images)

            caption_gt = captions
            dec_output = model(images, caption_gt)
            dec_output = dec_output[:, :-1].contiguous()
            captions_gt = caption_gt[:, 1:].contiguous()
            loss = loss_fn(dec_output.view(-1, dec_output.shape[-1]), captions_gt.view(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()
            # if scheduler is not None:
            #     scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss

def evaluate_loss(image_model, model, dataloader, loss_fn, text_field, epoch, device = 'cuda', args=None):
    # Validation loss
    model.eval()
    running_loss = .0

    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images, captions) in enumerate(dataloader):
                # if it == 10:
                #     break
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)
                if isinstance(captions, tuple) or isinstance(captions, list):
                    captions = [x.to(device) for x in captions]
                else:
                    captions = captions.to(device)

                if args.online:
                    images = image_model(images)

                caption_gt = captions
                dec_output = model(images, caption_gt)
                dec_output = dec_output[:, :-1].contiguous()
                captions_gt = caption_gt[:, 1:].contiguous()
                loss = loss_fn(dec_output.view(-1, dec_output.shape[-1]), captions_gt.view(-1))

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def train_scst(image_model, model, dataloader, optim, cider, text_field, epoch, device = 'cuda', scheduler = None, args=None):
    # Training with self-critical
    model.train()

    if scheduler is not None:
        scheduler.step()

    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            # if it == 2:
            #     break
            if isinstance(images, tuple) or isinstance(images, list):
                images = [x.to(device) for x in images]
                bs = images[0].shape[0]
            else:
                images = images.to(device)
                bs = images.shape[0]

            if args.online:
                with torch.no_grad():
                    images = image_model(images)

            outs, log_probs = model.beam_search(images, seq_len, text_field.vocab.stoi['<eos>'], beam_size, out_size=beam_size)

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(bs, beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)

            avg_log_probs = torch.sum(log_probs, -1) / torch.sum(log_probs != 0, -1)
            reward_loss = -avg_log_probs * (reward - reward_baseline)
            loss = reward_loss.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    return loss, reward


def evaluate_metrics(image_model, model, dataloader, text_field, epoch, device = 'cuda', args=None):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            # if it == 10:
            #     break
            with torch.no_grad():
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)
                
                if args.online:
                    images = image_model(images)
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores