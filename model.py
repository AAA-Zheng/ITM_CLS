
"""VSE model"""

import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from torch.nn.functional import max_pool1d, normalize
from thop import profile


class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.embed_size = opt.embed_size
        self.fc = nn.Linear(opt.img_dim, opt.embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        features = self.fc(images)
        features = features.permute(0, 2, 1)
        features = max_pool1d(features, features.size(2)).squeeze(2)
        features = normalize(features, p=2, dim=1)

        return features

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.embed_size = opt.embed_size
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, batch_first=True)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        ids = torch.LongTensor(lengths).cuda().view(-1, 1, 1)
        ids = ids.expand(x.size(0), 1, self.embed_size) - 1
        features = torch.gather(padded[0], 1, ids).squeeze(1)

        features = normalize(features, p=2, dim=1)

        return features
    

class RCLoss(nn.Module):

    def __init__(self, opt):
        super(RCLoss, self).__init__()

        self.label = opt.label
        self.margin = opt.margin
        self.gap = opt.gap
        self.batch_size = opt.batch_size
        self.pos_mask = torch.eye(self.batch_size).cuda()
        self.neg_mask = 1 - self.pos_mask

    def forward(self, v, t, image_emb, v_text_emb, t_text_emb):

        batch_size = v.size(0)

        scores = v.mm(t.t())
        pos_scores = scores.diag().view(batch_size, 1)
        pos_scores_t = pos_scores.expand_as(scores)
        pos_scores_v = pos_scores.t().expand_as(scores)

        if batch_size != self.batch_size:
            pos_mask = torch.eye(batch_size)
            pos_mask = pos_mask.cuda()
            neg_mask = 1 - pos_mask
        else:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        # triplet
        loss_t = (scores - pos_scores_t + self.margin).clamp(min=0)
        loss_v = (scores - pos_scores_v + self.margin).clamp(min=0)
        loss_t = loss_t * neg_mask
        loss_v = loss_v * neg_mask
        loss_t = loss_t.max(dim=1)[0]
        loss_v = loss_v.max(dim=0)[0]
        loss_t = loss_t.mean()
        loss_v = loss_v.mean()
        triplet_loss = (loss_t + loss_v) / 2

        # calculate relevance score
        if self.label == 'image':
            image_emb = image_emb.cuda()
            relevance = image_emb.mm(image_emb.t())
        else:
            v_text_emb = v_text_emb.cuda()
            t_text_emb = t_text_emb.cuda()

            v_text_emb = v_text_emb.transpose(0, 1)
            t_text_emb = t_text_emb.view(1, t_text_emb.size(0), t_text_emb.size(1))
            t_text_emb = t_text_emb.expand(5, t_text_emb.size(1), t_text_emb.size(2))

            relevance = torch.bmm(v_text_emb, t_text_emb.transpose(1, 2))
            relevance = relevance.max(0)[0]
        relevance = relevance * neg_mask + pos_mask

        # kendall
        relevance_repeat = relevance.unsqueeze(dim=2).repeat(1, 1, relevance.size(0))
        relevance_repeat_trans = relevance_repeat.permute(0, 2, 1)
        relevance_diff = (relevance_repeat_trans - relevance_repeat).clamp(max=0)
        neg_mask = torch.where(relevance_diff + self.gap < 0,
                               torch.full_like(relevance_diff, 1),
                               torch.full_like(relevance_diff, 0))
        num = neg_mask.sum()

        scores_repeat_t = scores.unsqueeze(dim=2).repeat(1, 1, scores.size(0))
        scores_repeat_trans_t = scores_repeat_t.permute(0, 2, 1)
        loss_t = scores_repeat_trans_t - scores_repeat_t

        scores_repeat_v = scores.t().unsqueeze(dim=2).repeat(1, 1, scores.size(0))
        scores_repeat_trans_v = scores_repeat_v.permute(0, 2, 1)
        loss_v = scores_repeat_trans_v - scores_repeat_v

        loss_t = loss_t.clamp(min=0)
        loss_v = loss_v.clamp(min=0)
        loss_t = loss_t * neg_mask
        loss_v = loss_v * neg_mask
        loss_t = loss_t.sum() / num
        loss_v = loss_v.sum() / num
        kendall_loss = (loss_t + loss_v) / 2

        loss = triplet_loss + kendall_loss

        return loss, triplet_loss, kendall_loss
    

class VSE(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt)
        self.txt_enc = EncoderText(opt)

        print(self.img_enc)
        print(self.txt_enc)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = RCLoss(opt)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        images = images.cuda()
        captions = captions.cuda()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)

        # flops, params = profile(self.img_enc, (images,))
        # print('Image encoder flops: %.2f G, params: %.2f M' % (flops / (10 ** 9), params / (10 ** 6)))
        # flops, params = profile(self.txt_enc, (captions, lengths,))
        # print('Text encoder flops: %.2f G, params: %.2f M' % (flops / (10 ** 9), params / (10 ** 6)))

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, image_emb, v_text_emb, t_text_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, loss_1, loss_2 = self.criterion(img_emb, cap_emb, image_emb, v_text_emb, t_text_emb)
        self.logger.update('L', loss.item(), img_emb.size(0))
        self.logger.update('L1', loss_1.item(), img_emb.size(0))
        self.logger.update('L2', loss_2.item(), img_emb.size(0))

        return loss

    def train_emb(self, images, captions, lengths, image_ids, caption_ids, image_emb, v_text_emb, t_text_emb, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, image_emb, v_text_emb, t_text_emb)

        # compute gradient and do SGD step
        loss.backward()
        clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
