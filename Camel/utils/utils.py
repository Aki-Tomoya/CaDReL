import itertools
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
# from PIL import Image
import torchvision.transforms as transforms
from PIL import Image


from common import evaluation

# def train_xe(image_model, target_model, online_model, discriminator_dec, discriminator_enc, dataloader, online_optim, target_optim, mlp_optim, loss_fn, text_field, epoch, online_scheduler, target_scheduler, mlp_scheduler, online_cider_list, target_cider_list, device = 'cuda', args=None):
def train_xe(image_model, target_model, online_model, discriminator_dec, discriminator_enc, dataloader, online_optim, target_optim, loss_fn, text_field, epoch, online_scheduler, target_scheduler, online_cider_list, target_cider_list, device = 'cuda', args=None):
     # Training with cross-entropy
    # kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    kl_loss_fn = nn.KLDivLoss(reduction="none")
    mse_loss_fn = nn.MSELoss(reduction="mean")
    msee_loss_fn = nn.MSELoss(reduction="none")
    bce_loss = nn.BCEWithLogitsLoss()
    # mse_loss_fn = nn.KLDivLoss(reduction="mean")
    target_model.train()
    online_model.train()
    discriminator_dec.train()
    online_scheduler.step()
    target_scheduler.step()
    # mlp_scheduler.step()
    # print('lr0 = ', optim.state_dict()['param_groups'][0]['lr'])
    running_loss = .0
    running_online_loss = .0
    running_distillation_loss = .0
    distillation_loss_parameter = .0
    mse_online_loss = .0
    device = args.device

    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions) in enumerate(dataloader):
            # if it == 10:
            #     break
            if isinstance(images, tuple) or isinstance(images, list):
                images = [x.to(device) for x in images]
            else:
                images = images.to(device)
            captions = captions.to(device)

            if args.online:
                with torch.no_grad():
                    images = image_model(images)
            
            online_logits, online_att, online_dec, online_difenc = online_model(images, captions, args)
            
            # online_logits, online_ml_outs = online_model(images, captions)
            online_text = F.log_softmax(online_logits, dim=-1)
            online_text = online_text[:, :-1].contiguous()
            captions_gt = captions[:, 1:].contiguous()

            online_loss = loss_fn(online_text.view(-1, online_text.shape[-1]), captions_gt.view(-1))
            
            
            #/w\


            # Knowledge distillation
            with torch.no_grad():
                target_logits, target_att, target_dec, target_difenc = target_model(images, captions, args)

            mse_loss = mse_loss_fn(online_att, target_att)

            # if epoch > 155 and target_cider_list[-1]-target_cider_list[-2]<0 and online_cider_list[-1]>target_cider_list[-1]:
            #     discrimination_loss=0

            # else:
            # Discrimitor_dec
            # enc_discriminator_input = torch.cat((online_difenc, target_difenc), dim=-1)
            discriminator_input = torch.cat((online_dec, target_dec), dim=-1)
            # discriminator_input = torch.cat((online_input, target_input), dim=-1)
            # discrimination = discriminator(discriminator_input)


            discrimination = discriminator_dec(discriminator_input)
            distillation_loss = kl_loss_fn(online_text, F.softmax(target_logits[:, :-1], dim=-1))
            distillation_loss = (discrimination[:, :-1] * distillation_loss).sum()
            distillation_loss = distillation_loss / online_text.size()[0]

            # old-loss
            # distillation_mseloss= ((online_logits - target_logits) ** 2).mean()

            # dis-enc-mseeloss
            # online_difenc = F.normalize(online_difenc, dim=-1)
            # target_difenc = F.normalize(target_difenc, dim=-1)
            # discrimination_enc = discriminator_enc(enc_discriminator_input)
            # discrimination_enc_loss = msee_loss_fn(online_difenc, target_difenc)
            # discrimination_enc_loss = (discrimination_enc[:, :] * discrimination_enc_loss).sum()
            # discrimination_enc_loss = discrimination_enc_loss / online_difenc.size()[0]

            #dis-enc-klloss
            # discrimination_enc = discriminator_enc(enc_discriminator_input)
            # discrimination_enc = 0.2 * discrimination_enc.expand(online_text.size()[0], online_text.size()[1], 1)
            # distillation_loss = (discrimination_enc * discrimination[:, :-1] * distillation_loss).sum()
            # distillation_loss = distillation_loss / online_text.size()[0]


            # camel-old loss
            distillation_weight = args.distillation_weight
            # distillation_loss = ((online_logits - target_logits) ** 2).mean()
            # distillation_loss *= distillation_weight
            loss = online_loss + distillation_loss + mse_loss
            # loss = online_loss + distillation_loss + mse_loss + distillation_loss_parameter * distillation_mseloss
            # loss = online_loss + distillation_loss + mse_loss + discrimination_enc_loss*0.0005
            # loss = online_loss + discrimination * distillation_loss + (1-discrimination) * discrimination_loss + mse_loss
            # loss = online_loss + distillation_loss_parameter * distillation_loss + (1-distillation_loss_parameter) * mse_loss

            online_optim.zero_grad()
            # mlp_optim.zero_grad()
            loss.backward()
            online_optim.step()
            # mlp_optim.step()

            # if scheduler is not None:
            #     scheduler.step()

            # EMA update
            with torch.no_grad():
                params_s = list(online_model.parameters())
                params_t = list(target_model.parameters())
                torch._foreach_mul_(params_t, args.ema_weight)
                w = torch._foreach_mul(params_s, 1 - args.ema_weight)
                torch._foreach_add_(params_t, w)





            #################################### update second #####################################
            if epoch > 10:
                target_logits, target_att, target_dec, target_difenc = target_model(images, captions, args)
                # target_logits, target_ml_outs = target_model(images, captions)
                target_text = F.log_softmax(target_logits, dim=-1)
                target_text = target_text[:, :-1].contiguous()
                captions_gt = captions[:, 1:].contiguous()

                target_loss = loss_fn(target_text.view(-1, target_text.shape[-1]), captions_gt.view(-1))

                # Knowledge distillation
                with torch.no_grad():
                    online_logits, online_att, online_dec, online_difenc = online_model(images, captions, args)
                    # online_logits, online_ml_outs = online_model(images, captions)   
                    
                mse_loss = mse_loss_fn(online_att, target_att)

                # if epoch > 299 and online_cider_list[epoch]-online_cider_list[epoch-1]<0 and target_cider_list[epoch]>online_cider_list[epoch]:
                #     discrimination_loss=0
                    
                # else:
                # Discriminator
                # enc_discriminator_input = torch.cat((online_difenc, target_difenc), dim=-1)
                discriminator_input = torch.cat((online_dec, target_dec), dim=-1)
                # discriminator_input = torch.cat((target_dec, online_dec), dim=-1)
                # discrimination = discriminator(discriminator_input)

                discrimination = discriminator_dec(discriminator_input)
                distillation_loss = kl_loss_fn(target_text, F.softmax(online_logits[:, :-1], dim=-1))
                distillation_loss = (discrimination[:, :-1] * distillation_loss).sum()
                distillation_loss = distillation_loss / target_text.size()[0]

                # old-loss
                # distillation_mseloss= ((online_logits - target_logits) ** 2).mean()

                #dis-enc-msee
                # online_difenc = F.normalize(online_difenc, dim=-1)
                # target_difenc = F.normalize(target_difenc, dim=-1)
                # discrimination_enc = discriminator_enc(enc_discriminator_input)
                # discrimination_enc_loss = msee_loss_fn(online_difenc, target_difenc)
                # discrimination_enc_loss = (discrimination_enc[:, :] * discrimination_enc_loss).sum()
                # discrimination_enc_loss = discrimination_enc_loss / online_difenc.size()[0]

                #dis-enc-mse
                # discrimination_enc = discriminator_enc(enc_discriminator_input)
                # discrimination_enc = 0.2 * discrimination_enc.expand(target_text.size()[0], target_text.size()[1], 1)
                # distillation_loss = (discrimination_enc * discrimination[:, :-1] * distillation_loss).sum()
                # distillation_loss = distillation_loss / target_text.size()[0]

                distillation_weight = args.distillation_weight
                # distillation_mseloss *= distillation_weight

                # loss = target_loss + distillation_loss
                loss = target_loss + distillation_loss + mse_loss
                # loss = target_loss + distillation_loss + mse_loss + distillation_loss_parameter * distillation_mseloss
                # loss = target_loss + distillation_loss + mse_loss + discrimination_enc_loss*0.0005
                # loss = target_loss + distillation_loss_parameter * distillation_loss + (1 - distillation_loss_parameter) * mse_loss

                target_optim.zero_grad()
                # mlp_optim.zero_grad()
                loss.backward()
                target_optim.step()
                # mlp_optim.step()


                # if scheduler is not None:
                #     scheduler.step()

                # EMA update
                with torch.no_grad():
                    params_t = list(online_model.parameters())
                    params_s = list(target_model.parameters())
                    torch._foreach_mul_(params_s, args.ema_weight)
                    w = torch._foreach_mul(params_t, 1 - args.ema_weight)
                    torch._foreach_add_(params_s, w)


            running_loss = running_loss + loss.item()
            running_online_loss = running_online_loss + online_loss.item()
            mse_online_loss = mse_online_loss + mse_loss.item()
            running_distillation_loss = running_distillation_loss + distillation_loss.item()
            pbar.set_postfix(online_loss=running_online_loss / (it + 1), 
                                distillation_loss=running_distillation_loss / (it + 1), 
                                mse_loss=mse_online_loss / (it + 1), 
                                 loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss



def evaluate_loss(image_model, target_model, online_model, dataloader, loss_fn, text_field, epoch, device = 'cuda', args=None):
    # Validation loss
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    target_model.eval()
    online_model.eval()
    running_loss = .0
    running_online_loss = .0
    running_distillation_loss = .0

    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images, captions) in enumerate(dataloader):
                # if it == 10:
                #     break
                captions = captions.to(device)
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)

                if args.online:
                    images = image_model(images)

                online_logits, online_att,  online_dec, online_difenc = online_model(images, captions, args)
                # online_logits, online_ml_outs = online_model(images, captions)
                online_text = F.log_softmax(online_logits, dim=-1)
                online_text = online_text[:, :-1].contiguous()
                captions_gt = captions[:, 1:].contiguous()

                online_loss = loss_fn(online_text.view(-1, online_text.shape[-1]), captions_gt.view(-1))

                target_logits, target_att, target_dec, target_difenc = target_model(images, captions, args)
                # target_logits, target_ml_outs = target_model(images, captions)

                distillation_loss = kl_loss_fn(online_text, F.softmax(target_logits[:, :-1], dim=-1))
                # distillation_loss = ((online_logits - target_logits) ** 2).mean()
                # distillation_loss = ((online_logits - target_logits) ** 2).mean() + ((online_ml_outs - target_ml_outs) ** 2).mean()
                distillation_weight = args.distillation_weight
                # distillation_loss *= distillation_weight

                loss = online_loss + distillation_loss

                running_loss = running_loss + loss.item()
                running_online_loss = running_online_loss + online_loss.item()
                running_distillation_loss = running_distillation_loss + distillation_loss.item()
                pbar.set_postfix(online_loss=running_online_loss / (it + 1), 
                                 distillation_loss=running_distillation_loss / (it + 1), 
                                 loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def train_scst(image_model, target_model, online_model, dataloader, optim, cider, text_field, epoch, device = 'cuda', scheduler = None, args=None):
    # Training with self-critical
    target_model.train()
    online_model.train()
    kl_loss_fn = nn.KLDivLoss(reduction="none")

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

            online_outs, online_log_probs, online_logits = online_model.beam_search(images, seq_len, text_field.vocab.stoi['<eos>'],
                                                                                    beam_size, out_size=beam_size, return_probs=True)

            # Rewards
            caps_gen = text_field.decode(online_outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(bs, beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)

            avg_online_log_probs = torch.sum(online_log_probs, -1) / torch.sum(online_log_probs != 0, -1)
            reward_loss = -avg_online_log_probs * (reward - reward_baseline)
            reward_loss = reward_loss.mean()
            loss = reward_loss

            # Knowledge distillation
            with torch.no_grad():
                target_outs, target_log_probs, target_logits = target_model.beam_search(images, seq_len, text_field.vocab.stoi['<eos>'],
                                                                                        beam_size, out_size=1, return_probs=True)

            best_target_logits = target_logits
            best_online_logits = online_logits[:, 0]

            mask = (best_online_logits == 0) | (best_target_logits == 0)
            best_online_logits.masked_fill_(mask, 0)
            best_target_logits.masked_fill_(mask, 0)

            distillation_loss = ((best_online_logits - best_target_logits) ** 2).mean()
            # distillation_loss = kl_loss_fn(best_online_logits, best_target_logits)
            distillation_weight = args.distillation_weight
            distillation_loss *= distillation_weight

            loss = loss + distillation_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            # EMA update
            with torch.no_grad():
                params_s = list(online_model.parameters())
                params_t = list(target_model.parameters())
                torch._foreach_mul_(params_t, args.ema_weight)
                w = torch._foreach_mul(params_s, 1 - args.ema_weight)
                torch._foreach_add_(params_t, w)

            running_loss = running_loss + loss.item()
            running_reward = running_reward + reward.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    return loss, reward


def evaluate_metrics(image_model, model, dataloader, text_field, epoch, device = 'cuda', args=None, reverse=False):
    import itertools
    import os

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

                # image_path = os.path.join('/home/public/caption/caption/coco2014/images/val2014', f'{it}.jpg') 
                # # image_path = image_paths[it]  # Get the complete image path corresponding to the current data item
                # # image = Image.open(image_path)

                if args.online:
                    images = image_model(images)
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)



            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                if reverse:
                    gen_i = list(reversed(gen_i))
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def train_xe_2(image_model, target_model, online_model, dataloader, online_optim, target_optim, loss_fn, text_field, epoch, online_scheduler, target_scheduler, device = 'cuda', args=None):
     # Training with cross-entropy
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    # mse_loss_fn = nn.KLDivLoss(reduction="mean")
    target_model.train()
    online_model.train()
    online_scheduler.step()
    target_scheduler.step()
    # print('lr0 = ', optim.state_dict()['param_groups'][0]['lr'])
    running_loss = .0
    running_online_loss = .0
    running_distillation_loss = .0
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

            lr_captions, rl_captions = captions[0], captions[1]

            if args.online:
                with torch.no_grad():
                    images = image_model(images)

            online_logits = online_model(images, lr_captions)
            online_text = F.log_softmax(online_logits, dim=-1)
            online_text = online_text[:, :-1].contiguous()
            captions_gt = lr_captions[:, 1:].contiguous()

            lens = torch.sum(captions_gt != 1, dim=-1)

            online_loss = loss_fn(online_text.view(-1, online_text.shape[-1]), captions_gt.view(-1))

            # Knowledge distillation
            with torch.no_grad():
                target_logits = target_model(images, rl_captions)[:, :-1]

            for i, l in enumerate(lens):
                target_logits[i,:l] = target_logits[i,:l].flip(-1)

            distillation_loss = kl_loss_fn(online_text[:, :-1], F.softmax(target_logits[:, 1:], dim=-1))
            # distillation_loss = ((online_logits - target_logits) ** 2).mean()
            # distillation_weight = args.distillation_weight
            # distillation_loss *= distillation_weight

            loss = online_loss + distillation_loss

            online_optim.zero_grad()
            loss.backward()
            online_optim.step()
            # if scheduler is not None:
            #     scheduler.step()

            # EMA update
            with torch.no_grad():
                params_s = list(online_model.parameters())
                params_t = list(target_model.parameters())
                torch._foreach_mul_(params_t, args.ema_weight)
                w = torch._foreach_mul(params_s, 1 - args.ema_weight)
                torch._foreach_add_(params_t, w)



            ##################################### update second #####################################
            target_logits = target_model(images, rl_captions)
            # target_logits, target_ml_outs = target_model(images, captions)
            target_text = F.log_softmax(target_logits, dim=-1)
            target_text = target_text[:, :-1].contiguous()
            captions_gt = rl_captions[:, 1:].contiguous()

            target_loss = loss_fn(target_text.view(-1, target_text.shape[-1]), captions_gt.view(-1))

            # Knowledge distillation
            with torch.no_grad():
                online_logits = online_model(images, lr_captions)[:, :-1]

            for i, l in enumerate(lens):
                online_logits[i,:l] = online_logits[i,:l].flip(-1)

            distillation_loss = kl_loss_fn(target_text[:, :-1], F.softmax(online_logits[:, 1:], dim=-1))
            # distillation_loss = ((target_logits - online_logits) ** 2).mean()
            distillation_weight = args.distillation_weight
            # distillation_loss *= distillation_weight

            loss = target_loss + distillation_loss

            target_optim.zero_grad()
            loss.backward()
            target_optim.step()
            # if scheduler is not None:
            #     scheduler.step()

            # EMA update
            with torch.no_grad():
                params_t = list(online_model.parameters())
                params_s = list(target_model.parameters())
                torch._foreach_mul_(params_s, args.ema_weight)
                w = torch._foreach_mul(params_t, 1 - args.ema_weight)
                torch._foreach_add_(params_s, w)


            running_loss = running_loss + loss.item()
            running_online_loss = running_online_loss + online_loss.item()
            running_distillation_loss = running_distillation_loss + distillation_loss.item()
            pbar.set_postfix(online_loss=running_online_loss / (it + 1), 
                                distillation_loss=running_distillation_loss / (it + 1), 
                                 loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss

def evaluate_loss_fn(model, dataloader, loss_fn, text_field, epoch, device = 'cuda', args=None):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images, captions) in enumerate(dataloader):
                # if it == 10:
                #     break
                captions = captions.to(device)
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)

                online_logits, online_att,  online_dec, online_difenc = model(images, captions, args)
                # online_logits, online_ml_outs = online_model(images, captions)
                online_text = F.log_softmax(online_logits, dim=-1)
                online_text = online_text[:, :-1].contiguous()

                captions = captions[:, 1:].contiguous()
                out = online_text
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

def discrimination_loss(discrimination, bce_loss, distillation_loss):
    
    labels = torch.ones_like(discrimination)
    loss = bce_loss(discrimination, labels)


    return loss

def train_scst_new(model, dataloader, optim, cider, text_field, epoch, device = 'cuda', scheduler = None, args=None):
    # Training with self-critical
    model.train()
    if scheduler is not None:
        scheduler.step()
    lr = optim.state_dict()['param_groups'][0]['lr']

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
            outs, log_probs = model.beam_search(images, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(bs, beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1), lr=lr)
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline