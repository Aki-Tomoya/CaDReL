import random
import torch
import time
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
import os
from shutil import copyfile

from common.utils.utils import setup_seed
from .utils import evaluate_loss, evaluate_metrics, train_scst, train_xe
# from torch.utils.tensorboard import SummaryWriter
from common.evaluation import PTBTokenizer, Cider
from common.data import DataLoader
from common.data.field import RawField

setup_seed(1234)

def train(args, image_model, model, datasets, image_field, text_field, optim=None, scheduler=None):

    device = args.device
    output = args.output
    use_rl = args.use_rl

    train_dataset, val_dataset, test_dataset = datasets

    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    cider_cache_path = 'cache/cider_cache.pkl'
    if use_rl:
        if os.path.exists(cider_cache_path):
            cider_train = torch.load(cider_cache_path)
        else:
            ref_caps_train = list(train_dataset.text)
            cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))            
            torch.save(cider_train, cider_cache_path)

        dict_train_dataset = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
        dict_dataloader_train = DataLoader(dict_train_dataset, batch_size=args.batch_size // 5, num_workers=args.workers, pin_memory=True)

    train_batch_size = args.batch_size // 5 if use_rl else args.batch_size
    dataloader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dataloader_val = DataLoader(val_dataset, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)

    def lambda_lr(s):
        if s <= 3:
            lr = args.xe_base_lr * s / 4
        elif s <= 10:
            lr = args.xe_base_lr
        elif s <= 12:
            lr = args.xe_base_lr * 0.2
        else:
            lr = args.xe_base_lr * 0.2 * 0.2
        return lr
    
    def lambda_lr_rl(s):
        refine_epoch = 28
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 3:
            lr = args.rl_base_lr * 0.2
        elif s <= refine_epoch + 6:
            lr = args.rl_base_lr * 0.2 * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
        return lr

    # Initial conditions
    if use_rl:
        optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
        scheduler = LambdaLR(optim, lambda_lr_rl)
    else:
        optim = AdamW(model.parameters(), lr=1, betas=(0.9, 0.98)) if optim is None else optim
        scheduler = LambdaLR(optim, lambda_lr) if scheduler is None else scheduler
    
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    best_cider = .0
    patience = 0
    start_epoch = 0

    last_saved = os.path.join(output, 'saved_models/%s_last.pth' % args.exp_name)
    best_saved = os.path.join(output, 'saved_models/%s_best.pth' % args.exp_name)

    if args.resume_last or args.resume_best:
        if use_rl:
            last_saved = os.path.join(output, 'saved_models/%s_rl_last.pth' % args.exp_name)
            best_saved = os.path.join(output, 'saved_models/%s_rl_best.pth' % args.exp_name)

        if args.resume_last:
            fname = last_saved
        else:
            fname = best_saved

        if os.path.exists(fname):
            data = torch.load(fname, map_location=device)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    elif use_rl:
        data = torch.load(best_saved, map_location=device)
        model.load_state_dict(data['state_dict'], strict=False)
        best_cider = data['best_cider']
        start_epoch = 0
        patience = 0
        print('Resuming from XE epoch %d, validation loss %f, and best cider %f' % (
            data['epoch'], data['val_loss'], data['best_cider']))

        last_saved = os.path.join(output, 'saved_models/%s_rl_last.pth' % args.exp_name)
        best_saved = os.path.join(output, 'saved_models/%s_rl_best.pth' % args.exp_name)

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        if not use_rl:
            train_loss = train_xe(image_model, model, dataloader_train, optim, loss_fn, text_field, e, device, scheduler, args)
        else:
            train_loss, reward = train_scst(image_model, model, dict_dataloader_train, optim, 
                                                             cider_train, text_field, e, device, scheduler, args)

        # Validation loss
        val_loss = evaluate_loss(image_model, model, dataloader_val, loss_fn, text_field, e, device, args)

        # Validation scores
        scores = evaluate_metrics(image_model, model, dict_dataloader_val, text_field, e, device, args)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']

        # Test scores
        scores = evaluate_metrics(image_model, model, dict_dataloader_test, text_field, e, device, args)
        print("Test scores", scores)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        # switch_to_rl = False
        exit_train = False
        # automatic training strategy 
        if patience == 5:
            if e < 15:
                patience = 0
            else:
                print('patience reached.')
                exit_train = True

        saved_dir = os.path.join(output, 'saved_models')
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, last_saved)

        if best:
            copyfile(last_saved, best_saved)

        if exit_train:
            break