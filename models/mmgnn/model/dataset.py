import json
import os
import random

import munch
import torch
import yaml
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoImageProcessor

from models.utils_finetune import AV2PT


class AVEDataset(Dataset):
    def __init__(self, config, tokenizer, image_processor, mode='train'):
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mode = mode
        self.qa_json_path = self.config.dataset.finetune_json_path.format(mode)
        self.data = json.load(open(self.qa_json_path, 'r', encoding='utf-8'))
        # if self.mode == 'train':
        #     self.data = shuffle(self.data)
        #     self.data = self.data[:len(self.data) // 2]
        self.gt_labels = ['Background', 'Church bell', 'Male speech, man speaking', 'Bark',
                          'Fixed-wing aircraft, airplane', 'Race car, auto racing', 'Female speech, woman speaking',
                          'Helicopter', 'Violin, fiddle', 'Flute', 'Ukulele', 'Frying (food)', 'Truck', 'Shofar',
                          'Motorcycle', 'Acoustic guitar', 'Train horn', 'Clock', 'Banjo', 'Goat',
                          'Baby cry, infant cry', 'Bus', 'Chainsaw', 'Cat', 'Horse', 'Toilet flush',
                          'Rodents, rats, mice', 'Accordion', 'Mandolin']
        self.av2pt = AV2PT(config, image_processor)
        self.use_augment = self.config.train.use_augment and self.mode == 'train'

    def sample_data(self, data):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        video_id = data['video_id']
        frame_id = data['frame_id']
        audio_name = data['audio_name']
        audio_path = os.path.join(self.config.dataset.data_path, video_id, audio_name)
        frame_name = data['frame_name']
        # frame_name = f'frame_{frame_id + 1}.jpg' # frame name + 1
        frame_path = os.path.join(self.config.dataset.data_path, video_id, frame_name)
        answer = data['answer']
        label = self.gt_labels.index(answer)
        return video_id, frame_id, audio_path, frame_path, answer, label

    def collate_fn(self, batch):
        video_ids, frame_ids, audio_paths, frame_paths, answers, labels = zip(*batch)
        questions = self.tokenizer(answers, padding=True, truncation=True, return_tensors='pt', max_length=77)
        text_input_ids = questions['input_ids']
        text_attention_mask = questions['attention_mask']
        labels = torch.tensor(labels)
        audio_feats, frame_feats = self.av2pt.av_to_pt(list(audio_paths), list(frame_paths), self.use_augment)

        # unbg_text_input_ids = [text_input_ids[i] for i in range(len(labels)) if labels[i] != 0]
        # unbg_text_input_ids = torch.stack(unbg_text_input_ids, dim=0)
        # unbg_text_attention_mask = [text_attention_mask[i] for i in range(len(labels)) if labels[i] != 0]
        # unbg_text_attention_mask = torch.stack(unbg_text_attention_mask, dim=0)
        return video_ids, frame_ids, text_input_ids, text_attention_mask, audio_feats, frame_feats, labels


if __name__ == '__main__':
    config = yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    config = munch.munchify(config)

    tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

    ave_dataset = AVEDataset(config, tokenizer, image_processor, mode='train')
    ave_dataloader = DataLoader(
        ave_dataset,
        batch_size=32,
        shuffle=True,
        # num_workers=4,
        # prefetch_factor=2,
        # persistent_workers=True,
        # pin_memory=True,
        drop_last=True,
        collate_fn=ave_dataset.collate_fn,
    )

    for data in tqdm(ave_dataloader):
        pass
