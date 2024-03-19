import torch
from tqdm import tqdm
import clip
from common import evaluation

def train_sm_xe(model, dataloader, optim, loss_fn, text_field, epoch, device = 'cuda', scheduler = None, args=None):
    # Training with cross-entropy
    model.train()
    if scheduler is not None:
        scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions) in enumerate(dataloader):
            images[0] = images[0].to(device)
            images[1] = images[1].to(device)
            images[2] = {
                    k1: {
                        k2: v2.to(device)
                        for k2, v2 in v1.items()
                    }
                    for k1, v1 in images[2].items()
                }
            captions = captions.to(device)

            out = model(images, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, out.shape[-1]), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            # if scheduler is not None:
            #     scheduler.step()

    loss = running_loss / len(dataloader)
    return loss

def evaluate_sm_loss(model, dataloader, loss_fn, text_field, epoch, device = 'cuda', args=None):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images, captions) in enumerate(dataloader):
                # if it == 10:
                #     break
                images[0] = images[0].to(device)
                images[1] = images[1].to(device)
                images[2] = {
                        k1: {
                            k2: v2.to(device)
                            for k2, v2 in v1.items()
                        }
                        for k1, v1 in images[2].items()
                    }
                captions = captions.to(device)

                out = model(images, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def train_xe(clip_model, model, dataloader, optim, loss_fn, text_field, epoch, device = 'cuda', scheduler = None, args=None):
    # Training with cross-entropy
    model.train()
    clip_model.eval()
    if scheduler is not None:
        scheduler.step()
    # print('lr0 = ', optim.state_dict()['param_groups'][0]['lr'])
    # print('lr1 = ', optim.state_dict()['param_groups'][1]['lr'])
    running_loss = .0
    running_cap_loss = .0
    running_sim_loss = .0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions) in enumerate(dataloader):
            # if it == 10:
            #     break
            captions = captions.to(device)
            if isinstance(images, tuple) or isinstance(images, list):
                images = [x.to(device) for x in images]
            else:
                images = images.to(device)
            out = model(images[:-1], captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            cap_loss = loss_fn(out.view(-1, out.shape[-1]), captions_gt.view(-1))

            out_max, caps_tokens = torch.max(out, dim=-1)
            with torch.no_grad():
                global_features = images[-1]
                caps_gen = text_field.decode(caps_tokens, join_words=False)
                caps_gen = [' '.join(cap).split('<unk>')[0] for cap in caps_gen]
                text_tokens = clip.tokenize(caps_gen).to(device)
                text_features = clip_model.encode_text(text_tokens).float()

                # global_features /= global_features.norm(dim=-1, keepdim=True)
                # text_features /= text_features.norm(dim=-1, keepdim=True)
                # similarity = (10 * global_features @ text_features.T)
                # reward = torch.diag(similarity)
                # reward_baseline = (similarity.sum(-1) - reward) / (similarity.shape[-1]-1)
                reward = torch.cosine_similarity(global_features, text_features)
                # reward = -torch.log(reward)

            # sim_loss = (-torch.mean(out_max, -1) * (reward - reward_baseline)).mean()
            sim_loss = (-torch.mean(out_max, -1) * reward).mean()

            loss = cap_loss + sim_loss
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_cap_loss += cap_loss.item()
            running_sim_loss += sim_loss.item()
            pbar.set_postfix(all_loss=running_loss / (it + 1), 
                                cap_loss=running_cap_loss / (it + 1),
                                sim_loss=running_sim_loss / (it + 1) 
                            )
            pbar.update()
            # if scheduler is not None:
            #     scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def evaluate_loss(clip_model, model, dataloader, loss_fn, text_field, epoch, device = 'cuda', args=None):
    # Validation loss
    model.eval()
    clip_model.eval()
    running_loss = .0
    running_cap_loss = .0
    running_sim_loss = .0
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
                out = model(images[:-1], captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                cap_loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))

                global_features = images[-1]
                out_max, caps_tokens = torch.max(out, dim=-1)
                caps_gen = text_field.decode(caps_tokens, join_words=False)
                caps_gen = [' '.join(cap).split('<unk>')[0] for cap in caps_gen]
                text_tokens = clip.tokenize(caps_gen).to(device)
                text_features = clip_model.encode_text(text_tokens).float()

                # global_features /= global_features.norm(dim=-1, keepdim=True)
                # text_features /= text_features.norm(dim=-1, keepdim=True)
                # similarity = (10 * global_features @ text_features.T)
                # reward = torch.diag(similarity)
                # reward_baseline = (similarity.sum(-1) - reward) / (similarity.shape[-1]-1)
                reward = torch.cosine_similarity(global_features, text_features)
                reward = -torch.log(reward)

                # sim_loss = (-torch.mean(out_max, -1) * (reward - reward_baseline)).mean()
                sim_loss = (-torch.mean(out_max, -1) * reward).mean()

                loss = cap_loss + sim_loss
                running_loss += loss.item()
                running_cap_loss += cap_loss.item()
                running_sim_loss += sim_loss.item()
                pbar.set_postfix(all_loss=running_loss / (it + 1), 
                                    cap_loss=running_cap_loss / (it + 1),
                                    sim_loss=running_sim_loss / (it + 1) 
                                )
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss