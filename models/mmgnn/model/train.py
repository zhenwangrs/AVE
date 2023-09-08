# coding=utf8
import logging
import os
import random
import warnings

import munch
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AVEDataset
from model import AVE_Model_TF as AVE_Model

warnings.filterwarnings('ignore')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# torch.set_float32_matmul_precision('high')


def train():
    logger.info(config)

    model = AVE_Model(config).to(config.train.device)
    torch.save(model.state_dict(), f'./ckp/model_0.pth')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    scaler = GradScaler()

    optim = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    # optim = torch.optim.SGD(model.parameters(), lr=config.train.lr)
    # optim = torch.optim.AdamW(model.parameters(), lr=config.train.lr, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.009)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=config.train.epochs)

    start_epoch = config.train.start_epoch
    if config.train.start_epoch > 1:
        model.load_state_dict(torch.load(f'./ckp/model_{start_epoch - 1}.pth'))

    for i in range(start_epoch - 1):
        scheduler.step()

    if config.train.training_mode:
        train_dataset = AVEDataset(config, model.tokenizer, model.image_processor, 'train')
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.train.batch_size,
                                  shuffle=True,
                                  num_workers=6,
                                  prefetch_factor=2,
                                  persistent_workers=True,
                                  pin_memory=True,
                                  collate_fn=train_dataset.collate_fn,
                                  drop_last=True)

        test_dataset = AVEDataset(config, model.tokenizer, model.image_processor, 'val')
        test_loader = DataLoader(test_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 num_workers=4,
                                 prefetch_factor=2,
                                 persistent_workers=True,
                                 pin_memory=True,
                                 collate_fn=test_dataset.collate_fn,
                                 drop_last=False)

        best_acc = 0
        for epoch in range(start_epoch, config.train.epochs + 1):
            epoch_loss = 0
            log_freq = len(train_loader) // config.train.log_freq
            for index, (video_ids, frame_ids, unbg_text_input_ids, unbg_text_attention_mask, audio_feats, frame_feats, labels) in enumerate(
                tqdm(train_loader), start=1):
                unbg_text_input_ids = unbg_text_input_ids.to(config.train.device)
                unbg_text_attention_mask = unbg_text_attention_mask.to(config.train.device)
                audio_feats = audio_feats.to(config.train.device)
                frame_feats = frame_feats.to(config.train.device)
                labels = labels.to(config.train.device)

                with autocast():
                    loss, cls_loss, cosine_loss, clip_loss, logits = model(unbg_text_input_ids, unbg_text_attention_mask, audio_feats, frame_feats, labels)
                scaler.scale(loss).backward()
                epoch_loss += loss.item()
                print(f'epoch: {epoch}, batch: {index}, loss: {loss.item()}, cls_loss: {cls_loss.item()}, '
                      f'cosine_loss: {cosine_loss.item()}, clip_loss: {clip_loss.item()}, epoch_loss: {epoch_loss / index}')
                if index % config.train.batch_accum == 0 or index == len(train_loader):
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

                if index % log_freq == 0:
                    logger.info(f'epoch: {epoch}, batch: {index}, loss: {loss.item()}, cls_loss: {cls_loss.item()}, '
                                f'cosine_loss: {cosine_loss.item()}, clip_loss: {clip_loss.item()}, epoch_loss: {epoch_loss / index}')

                torch.cuda.empty_cache()
            logger.info(f'epoch loss: {epoch_loss / index}')

            scheduler.step()

    logger.info('testing ...')
    model.load_state_dict(torch.load(f'./ckp/model_best.pth'), strict=False)
    test(model, config, split='test')


def test(model, config, split='test', test_loader=None):
    model.eval()

    if test_loader is None:
        test_dataset = AVEDataset(config, model.tokenizer, model.image_processor, split)
        test_loader = DataLoader(test_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 num_workers=4,
                                 prefetch_factor=2,
                                 persistent_workers=False,
                                 pin_memory=True,
                                 collate_fn=test_dataset.collate_fn,
                                 drop_last=False)

    with torch.no_grad():
        answer_record = {}
        for index, (video_ids, frame_ids, unbg_text_input_ids, unbg_text_attention_mask, audio_feats, frame_feats, labels) in enumerate(
            tqdm(test_loader), start=1):
            unbg_text_input_ids = unbg_text_input_ids.to(config.train.device)
            unbg_text_attention_mask = unbg_text_attention_mask.to(config.train.device)
            audio_feats = audio_feats.to(config.train.device)
            frame_feats = frame_feats.to(config.train.device)
            labels = labels.to(config.train.device)
            logits = model.predict(unbg_text_input_ids, unbg_text_attention_mask, audio_feats, frame_feats, labels)
            pred_score_list = torch.softmax(logits, dim=-1).cpu().tolist()
            for vid, fid, pred_score, label in zip(video_ids, frame_ids, pred_score_list, labels):
                if vid not in answer_record:
                    answer_record[vid] = []

                answer_record[vid].append({
                    'frame_id': fid,
                    'pred_ans_score': pred_score,
                    'pred_ans_index': np.argmax(pred_score),
                    'correct_ans_index': label,
                })

    # 计算准确率
    total_correct = 0
    total_qa = 0
    for vid, record in answer_record.items():
        correct_list = []
        pred_list = []
        frames = sorted(record, key=lambda x: x['frame_id'])
        for frame in frames:
            correct_list.append(frame['correct_ans_index'].item())
            pred_list.append(frame['pred_ans_index'].item())
            total_qa += 1
            if frame['correct_ans_index'] == np.argmax(frame['pred_ans_score']):
                total_correct += 1
        print(f'{vid} correct: {correct_list}, pred: {pred_list}')
    logger.info(
        f'{split} total correct: {total_correct}, total: {total_qa}, acc: {total_correct / total_qa}')

    def post_process_1(pred_list):
        # 取出数组中非0的最多的那个数字，将所有非0的数字替换为这个数字
        pred_list = np.array(pred_list)
        nonzero_pred_list = pred_list[pred_list != 0]
        if len(nonzero_pred_list) == 0:
            return pred_list
        count = np.bincount(nonzero_pred_list)
        pred_category = np.argmax(count)
        num = np.sum(count == count[pred_category])
        # if num == 1:
        for i in range(len(pred_list)):
            if pred_list[i] != 0:
                pred_list[i] = pred_category
        return pred_list

    def post_process_2(pred_list):
        # 取出数组中非0的最多的那个数字，将所有非0的数字替换为这个数字
        pred_list = np.array(pred_list)
        nonzero_pred_list = pred_list[pred_list != 0]
        if len(nonzero_pred_list) == 0:
            return pred_list
        count = np.bincount(nonzero_pred_list)
        pred_category = np.argmax(count)
        num = np.sum(count == count[pred_category])
        # if num == 1:
        for i in range(len(pred_list)):
            if pred_list[i] != 0:
                pred_list[i] = pred_category
        # else:
        #     print(f'num: {num}, count: {count}, pred_category: {pred_category}')

        # 遍历数组，遇到第一个非0的数字，将其后面的数字全部替换为这个数字，直到遇到下一个0
        # 找到第一个非0的数字的下标
        first_nonzero_index = -1
        for i in range(len(pred_list)):
            if pred_list[i] != 0:
                first_nonzero_index = i
                break
        if first_nonzero_index == -1:
            return pred_list
        # 找到最后一个非0的数字的下标
        last_nonzero_index = -1
        for i in range(len(pred_list) - 1, -1, -1):
            if pred_list[i] != 0:
                last_nonzero_index = i
                break
        if last_nonzero_index == -1:
            return pred_list
        # 将第一个非0的数字后面的数字全部替换为pred_category
        for i in range(first_nonzero_index, last_nonzero_index + 1):
            pred_list[i] = pred_category
        return pred_list

    # 计算准确率
    total_correct = 0
    total_qa = 0
    for vid, record in answer_record.items():
        correct_list = []
        pred_list = []
        frames = sorted(record, key=lambda x: x['frame_id'])
        for frame in frames:
            correct_list.append(frame['correct_ans_index'].item())
            pred_list.append(frame['pred_ans_index'].item())
        pred_list = post_process_1(pred_list)
        for i in range(len(pred_list)):
            total_qa += 1
            if pred_list[i] == correct_list[i]:
                total_correct += 1
    acc = total_correct / total_qa
    logger.info(f'{split} total correct: {total_correct}, total: {total_qa}, acc: {acc}')

    # 计算准确率
    total_correct = 0
    total_qa = 0
    for vid, record in answer_record.items():
        correct_list = []
        pred_list = []
        frames = sorted(record, key=lambda x: x['frame_id'])
        for frame in frames:
            correct_list.append(frame['correct_ans_index'].item())
            pred_list.append(frame['pred_ans_index'].item())
        pred_list = post_process_2(pred_list)
        for i in range(len(pred_list)):
            total_qa += 1
            if pred_list[i] == correct_list[i]:
                total_correct += 1
    acc = total_correct / total_qa
    logger.info(f'{split} total correct: {total_correct}, total: {total_qa}, acc: {acc}')

    model.train()
    return acc


if __name__ == '__main__':
    logging.basicConfig(filename='log.log', level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('t2i2p')
    logger.info('music avqa')
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(chlr)

    config = yaml.load(open('config.yaml', 'r', encoding='utf-8'),
                       Loader=yaml.FullLoader)
    config = munch.munchify(config)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.train.visible_gpu
    train()
