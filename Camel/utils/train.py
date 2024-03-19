import random
import torch
import time
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from torch import nn
import torch.nn.functional as F
import os
from shutil import copyfile
from Camel.utils.utils import evaluate_loss, evaluate_metrics, train_scst, train_xe, train_xe_2, evaluate_loss_fn, train_scst_new
# from torch.utils.tensorboard import SummaryWriter
from common.evaluation import PTBTokenizer, Cider
from common.data import DataLoader
from common.data.field import RawField
from torch.nn.utils.rnn import pad_sequence
from Camel.utils.lookahead import Lookahead

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def train(args, image_model, target_model, online_model, datasets, image_field, text_field, optim=None, scheduler=None):

    device = args.device
    output = args.output
    use_rl = args.use_rl

    date = time.strftime("%Y-%m-%d", time.localtime())

    train_dataset, val_dataset, test_dataset = datasets

    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    cider_cache_path = '/home/zkx/ImgCap/CaDReL/captioning/cache/cider_cache.pkl'
    
    if use_rl:
        if os.path.exists(cider_cache_path):
            cider_train = torch.load(cider_cache_path)
        else:
            ref_caps_train = list(train_dataset.text)
            cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))            
            torch.save(cider_train, cider_cache_path)

        train_dataset = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    train_batch_size = args.batch_size // 5 if use_rl else args.batch_size
    dataloader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dataloader_val = DataLoader(val_dataset, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)

    # def lambda_lr(s):
    #     warm_up = args.warmup
    #     s += 1
    #     lr = (online_model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    #     return lr

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
        refine_epoch = 8
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 3:
            lr = args.rl_base_lr * 0.2
        elif s <= refine_epoch + 6:
            lr = args.rl_base_lr * 0.2 * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
        return lr
    
    def lambda_mlp_lr(s):
        if s <= 2:
            lr = (args.xe_base_lr * s / 4) 
        elif s <= 5:
            lr = (args.xe_base_lr)
        elif s <= 10:
            lr = (args.xe_base_lr * 0.2) 
        else:
            lr = (args.xe_base_lr * 0.2 * 0.2) 
        return lr
    
    # Initial Discrimitor
    discriminator_dec = Discriminator().to(device)
    if args.use_premlp :
        mlp_data = torch.load('/home/zkx/ImgCap/CaDReL/captioning/Camel/saved/mlp126.11.pth')
        discriminator_dec.load_state_dict(mlp_data)

    discriminator_enc = Discriminator_enc().to(device)

    # Initial conditions
    if use_rl:
        optim_rl = Adam(online_model.parameters(), lr=1, betas=(0.9, 0.98))
        scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
    else:
        online_optim = AdamW(online_model.parameters(), lr=1, betas=(0.9, 0.98))
        target_optim = AdamW(target_model.parameters(), lr=1, betas=(0.9, 0.98))
        # online_optim = AdamW([
        #         {'params': online_model.parameters()},
        #         {'params': discriminator_dec.parameters(), 'lr': 1e-4}
        #     ], lr=1, betas=(0.9, 0.98))
        # target_optim = AdamW([
        #         {'params': online_model.parameters()},
        #         {'params': discriminator_dec.parameters(), 'lr': 1e-4}
        #     ], lr=1, betas=(0.9, 0.98))
        # mlp_optim = Lion(discriminator_dec.parameters(), lr=0.002, betas=(0.95, 0.98))
        mlp_optim = AdamW(discriminator_dec.parameters(), lr=1, betas=(0.9, 0.98))
        # mlp_optim = Lookahead(mlp_optim, la_steps=5, la_alpha=0.8)
        online_scheduler = LambdaLR(online_optim, lambda_lr)
        target_scheduler = LambdaLR(target_optim, lambda_lr)
        mlp_scheduler = LambdaLR(mlp_optim, lambda_mlp_lr)
        # mlp_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(mlp_optim, T_0=12, T_mult=2, eta_min=0.0001)


    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    best_cider = .0
    patience = 0
    start_epoch = 0

    last_saved = os.path.join(output, 'saved_models3/%s_last.pth' % args.exp_name)
    best_saved = os.path.join(output, 'saved_models3/%s_best.pth' % args.exp_name)
    mlp_saved = os.path.join(output, 'saved_models3/%s_mlp.pth' % args.exp_name)

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = last_saved
        else:
            fname = best_saved

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            target_model.load_state_dict(data['state_dict_t'], strict=False)
            online_model.load_state_dict(data['state_dict_o'], strict=False)
            target_optim.load_state_dict(data['optimizer_t'])
            online_optim.load_state_dict(data['optimizer_o'])
            target_scheduler.load_state_dict(data['scheduler_t'])
            online_scheduler.load_state_dict(data['scheduler_o'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))


    print("Training starts")
    print(best_saved)
    online_cider_list=[]
    target_cider_list=[]
    for e in range(start_epoch, start_epoch + 100):
        if not use_rl:
            # train_loss = train_xe(image_model, target_model, online_model, discriminator_dec, discriminator_enc, dataloader_train, online_optim, target_optim, mlp_optim, loss_fn, text_field, e, online_scheduler, target_scheduler, mlp_scheduler, online_cider_list, target_cider_list, device, args)
            train_loss = train_xe(image_model, target_model, online_model, discriminator_dec, discriminator_enc, dataloader_train, online_optim, target_optim, loss_fn, text_field, e, online_scheduler, target_scheduler, online_cider_list, target_cider_list, device, args)
        else:
            # train_loss, reward = train_scst(image_model, target_model, online_model, dataloader_train, optim_rl, 
            #                                                  cider_train, text_field, e, device, scheduler_rl, args)
            train_loss, reward = train_scst_new(online_model, dataloader_train, optim_rl, 
                                                             cider_train, text_field, e, device, scheduler_rl, args)

        # Validation loss
        val_loss = evaluate_loss(image_model, target_model, online_model, dataloader_val, loss_fn, text_field, e, device, args)

        # Validation scores
        scores = evaluate_metrics(image_model, online_model, dict_dataloader_test, text_field, e, device, args)
        print("Online test scores", scores)
        val_cider = scores['CIDEr']
        online_cider_list.append(val_cider)

        # Test scores
        scores = evaluate_metrics(image_model, target_model, dict_dataloader_test, text_field, e, device, args, reverse=False)
        print("Target test scores", scores)
        target_cider_list.append(scores['CIDEr'])

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
            # if e < 15:
            if e < 30:
                patience = 0
            else:
                print('patience reached.')
                exit_train = True

        saved_dir = os.path.join(output, 'saved_models3')
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        
        if use_rl:
             torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'val_loss': val_loss,
                'val_cider': val_cider,
                'state_dict_t': target_model.state_dict(),
                'state_dict_o': online_model.state_dict(),
                'optimizer': optim_rl.state_dict(),
                'scheduler': scheduler_rl.state_dict() if scheduler else None,
                'patience': patience,
                'best_cider': best_cider,
                'use_rl': use_rl,
            }, last_saved)

        
        else:
            torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'val_loss': val_loss,
                'val_cider': val_cider,
                'state_dict_t': target_model.state_dict(),
                'state_dict_o': online_model.state_dict(),
                'optimizer_t': target_optim.state_dict(),
                'optimizer_o': online_optim.state_dict(),
                'scheduler_t': target_scheduler.state_dict(),
                'scheduler_o': online_scheduler.state_dict(),
                'patience': patience,
                'best_cider': best_cider,
                'use_rl': use_rl,
            }, last_saved)

            

        if best:
            copyfile(last_saved, best_saved)
            try:
                torch.save(discriminator_dec.state_dict(), mlp_saved)
            except MemoryError:
                print('mlp not saved!')

        if exit_train:
            # writer.close()
            break

def train_s(args, image_model, online_model, datasets, image_field, text_field, optim=None, scheduler=None):
    device = args.device
    output = args.output
    use_rl = args.use_rl
    use_srl = args.use_srl

    date = time.strftime("%Y-%m-%d", time.localtime())

    train_dataset, val_dataset, test_dataset = datasets

    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    cider_cache_path = '/home/zkx/ImgCap/CaDReL/captioning/cache/cider_cache.pkl'
    
    if use_srl:
        if os.path.exists(cider_cache_path):
            cider_train = torch.load(cider_cache_path)
        else:
            ref_caps_train = list(train_dataset.text)
            cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))            
            torch.save(cider_train, cider_cache_path)

        train_dataset = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    train_batch_size = args.batch_size // 5 if use_srl else args.batch_size
    dataloader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dataloader_val = DataLoader(val_dataset, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)

    # def lambda_lr(s):
    #     warm_up = args.warmup
    #     s += 1
    #     lr = (online_model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    #     return lr

    
    def lambda_lr_rl(s):
        refine_epoch = 8
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 3:
            lr = args.rl_base_lr * 0.2
        elif s <= refine_epoch + 6:
            lr = args.rl_base_lr * 0.2 * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
        return lr
    
    def lambda_mlp_lr(s):
        if s <= 2:
            lr = (args.xe_base_lr * s / 4) 
        elif s <= 5:
            lr = (args.xe_base_lr)
        elif s <= 10:
            lr = (args.xe_base_lr * 0.2) 
        else:
            lr = (args.xe_base_lr * 0.2 * 0.2) 
        return lr
    
    # Initial Discrimitor
    discriminator_dec = Discriminator().to(device)
    if args.use_premlp :
        mlp_data = torch.load('/home/zkx/ImgCap/CaDReL/captioning/Camel/saved/mlp126.11.pth')
        discriminator_dec.load_state_dict(mlp_data)

    discriminator_enc = Discriminator_enc().to(device)

    # Initial conditions
    if use_srl:
        optim_rl = Adam(online_model.parameters(), lr=1, betas=(0.9, 0.98))
        scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
    else:
        online_optim = AdamW(online_model.parameters(), lr=1, betas=(0.9, 0.98))

       


    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    best_cider = .0
    patience = 0
    start_epoch = 0

    last_saved = os.path.join(output, 'saved_models4/%s_last.pth' % args.exp_name)
    best_saved = os.path.join(output, 'saved_models4/%s_best.pth' % args.exp_name)
    mlp_saved = os.path.join(output, 'saved_models4/%s_mlp.pth' % args.exp_name)


    print("Training starts")
    print(best_saved)
    online_cider_list=[]
    target_cider_list=[]
    for e in range(start_epoch, start_epoch + 100):
        if not use_srl:
            train_loss = train_xe(image_model,online_model, online_model, discriminator_dec, discriminator_enc, dataloader_train, online_optim, optim_rl, optim_rl, loss_fn, text_field, e, scheduler_rl, scheduler_rl, scheduler_rl, online_cider_list, target_cider_list, device, args)
            
        else:
            # train_loss, reward = train_scst(image_model, target_model, online_model, dataloader_train, optim_rl, 
            #                                                  cider_train, text_field, e, device, scheduler_rl, args)
            train_loss, reward, baseline_reward = train_scst_new(online_model, dataloader_train, optim_rl, 
                                                             cider_train, text_field, e, device, scheduler_rl, args)

        # Validation loss
        val_loss = evaluate_loss_fn(online_model, dataloader_val, loss_fn, text_field, e, device, args)

        # Validation scores
        scores = evaluate_metrics(image_model, online_model, dict_dataloader_test, text_field, e, device, args)
        print("Online test scores", scores)
        val_cider = scores['CIDEr']
        online_cider_list.append(val_cider)


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
            # if e < 15:
            if e < 30:
                patience = 0
            else:
                print('patience reached.')
                exit_train = True

        saved_dir = os.path.join(output, 'saved_models4')
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        
        if use_srl:
             torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'val_loss': val_loss,
                'val_cider': val_cider,
                # 'state_dict_t': target_model.state_dict(),
                'state_dict_o': online_model.state_dict(),
                'optimizer': optim_rl.state_dict(),
                'scheduler': scheduler_rl.state_dict() if scheduler else None,
                'patience': patience,
                'best_cider': best_cider,
                'use_rl': use_rl,
            }, last_saved)

        
        else:
            torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'val_loss': val_loss,
                'val_cider': val_cider,
                # 'state_dict_t': target_model.state_dict(),
                'state_dict_o': online_model.state_dict(),
                # 'optimizer_t': target_optim.state_dict(),
                'optimizer_o': online_optim.state_dict(),
                # 'scheduler_t': target_scheduler.state_dict(),
                # 'scheduler_o': online_scheduler.state_dict(),
                'patience': patience,
                'best_cider': best_cider,
                'use_rl': use_rl,
            }, last_saved)

            

        if best:
            copyfile(last_saved, best_saved)
            try:
                torch.save(discriminator_dec.state_dict(), mlp_saved)
            except MemoryError:
                print('mlp not saved!')

        if exit_train:
            # writer.close()
            break

# Discrimitor
class Discriminator(nn.Module):
    # old:d_out=256, d_model=512, d_in = 1024
    # new:d_out=512, d_model=2048, d_in = 1024
    # Trans:d_out=512, d_model=4096, d_in=1024, dropout=0.
    # d_end=64, d_last=128, d_pre=256, d_out=512, d_mid=2048, d_model=4096, d_in=1024,
    # d_hid6=8, d_end=16, d_last=32, d_pre=64, d_out=128, d_mid=256, d_model=512,
    # d_layer7=32, d_layer6=64, d_layer5=128, d_layer4=256, d_layer3=512, d_layer2=1024, d_layer1=4096, d_in=1024,
    def __init__(self,d_layer7=9, d_layer6=16, d_layer5=32, d_layer4=64, d_layer3=128, d_layer2=256, d_layer1=512, d_in=1024, dropout=0.5):
        super(Discriminator, self).__init__()
        self.hidden1 = nn.Linear(d_in, d_layer1)
        self.hidden2 = nn.Linear(d_layer1, d_layer2)
        self.hidden3 = nn.Linear(d_layer2, d_layer3)
        self.hidden4 = nn.Linear(d_layer3, d_layer4)
        # self.hidden5 = nn.Linear(d_layer4, d_layer5)
        # self.hidden6 = nn.Linear(d_layer5, d_layer6)
        self.predict = nn.Linear(d_layer4, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
    
    def forward(self, input):
        x = self.hidden1(input)
        # x = self.relu(x)
        x = self.gelu(x)
        # x = self.dropout(x)

        x = self.hidden2(x)
        # x = self.relu(x)
        x = self.gelu(x)
        # x = self.dropout(x)

        x = self.hidden3(x)
        # x = self.relu(x)
        x = self.gelu(x)
        # x = self.dropout(x)

        x = self.hidden4(x)
        # x = self.relu(x)
        x = self.gelu(x)
        # x = self.dropout(x)

        # x = self.hidden5(x)
        # x = self.relu(x)
        # x = self.gelu(x)
        # x = self.dropout(x)

        # x = self.hidden6(x)
        # x = self.relu(x)

        x = self.predict(x)
        x = self.sigmoid(x)

        return x
    
# Discrimitor
class Discriminator2(nn.Module):
    # old:d_out=256, d_model=512, d_in = 1024
    # new:d_out=512, d_model=2048, d_in = 1024
    # Trans:d_out=512, d_model=4096, d_in=1024, dropout=0.
    # d_end=64, d_last=128, d_pre=256, d_out=512, d_mid=2048, d_model=4096, d_in=1024,
    # d_hid6=8, d_end=16, d_last=32, d_pre=64, d_out=128, d_mid=256, d_model=512,
    # d_layer7=32, d_layer6=64, d_layer5=128, d_layer4=256, d_layer3=512, d_layer2=1024, d_layer1=4096, d_in=1024,
    def __init__(self,d_hid6=32, d_end=64, d_last=128, d_pre=256, d_out=512, d_mid=1024, d_model=4096, d_in=1024, dropout=0.5):
        super(Discriminator, self).__init__()
        self.hidden1 = nn.Linear(d_in, d_model)
        self.hidden_mid = nn.Linear(d_model, d_mid)
        self.hidden2 = nn.Linear(d_model, d_out)
        self.hidden3 = nn.Linear(d_out, d_pre)
        self.hidden4 = nn.Linear(d_pre, d_last)
        self.hidden5 = nn.Linear(d_last, d_end)
        # self.hidden6 = nn.Linear(d_end, d_hid6)
        self.predict = nn.Linear(d_end, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
    
    def forward(self, input):
        x = self.hidden1(input)
        x = self.relu(x)
        # x = self.hidden_mid(x)
        # x = self.elu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.hidden4(x)
        x = self.relu(x)
        x = self.hidden5(x)
        x = self.relu(x)
        # x = self.hidden6(x)
        # x = self.relu(x)

        x = self.predict(x)
        x = self.sigmoid(x)

        return x

class Discriminator_enc(nn.Module):
    # old:d_out=256, d_model=512, d_in = 1024
    # new:d_out=512, d_model=2048, d_in = 1024
    # Trans:d_out=512, d_model=4096, d_in=1024, dropout=0.
    # 6x :
    def __init__(self, d_last=128, d_pre=256, d_out=512, d_mid=2048, d_model=4096, d_in=1024, dropout=0.5):
        super(Discriminator_enc, self).__init__()
        self.hidden1 = nn.Linear(d_in, d_model)
        self.hidden_mid = nn.Linear(d_model, d_mid)
        self.hidden2 = nn.Linear(d_model, d_out)
        self.hidden3 = nn.Linear(d_out, d_pre)
        self.hidden4 = nn.Linear(d_pre, d_last)
        self.predict = nn.Linear(d_last, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(20, 81, 1)
        self.relu = nn.ReLU()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, ), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU()
        )
    
    def forward(self, input):
        x = self.hidden1(input)
        x = self.relu(x)
        # x = self.hidden_mid(x)
        # x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.hidden4(x)
        x = self.relu(x)
        x = self.predict(x)
        x = self.sigmoid(x)

        #注意力+卷积
        x = torch.mean(x, axis=1, keepdims=True)
        return x