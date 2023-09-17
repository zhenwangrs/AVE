import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from transformers import AutoTokenizer, CLIPVisionModel, AutoProcessor

from models.mmgnn.model import models_audio_mae
from models.mmh_vig import mm_vig
from open_clip import ClipLoss


class PairwiseCrossAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super(PairwiseCrossAttention, self).__init__()
        self.cma = nn.MultiheadAttention(hidden_size, 8, batch_first=True, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, audio_feature, vision_feature):
        attended_values = self.cma(audio_feature, vision_feature, vision_feature)[0]
        attended_values = self.layer_norm1(attended_values)
        attended_values = self.FFN(attended_values)
        attended_values = self.layer_norm2(attended_values)
        # attended_values = torch.cat([audio_feature, vision_feature], dim=1)
        return attended_values


class AVE_Model_TF(nn.Module):
    def __init__(self, config):
        super(AVE_Model_TF, self).__init__()
        self.hidden_size = 192

        self.mmvig = mm_vig.vig_ti16_224_gelu()
        self.classifier = nn.Linear(1000 + self.hidden_size * 2, 29)

        self.config = config
        self.audio_mae = models_audio_mae.__dict__[config.model.model_name](
            norm_pix_loss=config.model.norm_pix_loss,
            in_chans=config.model.in_channels,
            audio_exp=config.model.audio_exp,
            img_size=(config.dataset.target_length, 128),
            alpha=config.model.alpha,
            mode=config.model.mode,
            use_custom_patch=config.model.use_custom_patch,
            split_pos=config.model.split_pos,
            pos_trainable=config.model.pos_trainable,
            use_nce=config.model.use_nce,
            decoder_mode=config.model.decoder_mode,
            mask_2d=config.model.mask_2d,
            mask_t_prob=config.model.mask_t_prob,
            mask_f_prob=config.model.mask_f_prob,
            no_shift=config.model.no_shift,
        )
        self.audio_mae.load_state_dict(torch.load(config.model.pretrained_audio_mae_path))
        self.set_if_finetune(self.audio_mae, config.model.finetune.audio_mae)
        self.audio_mae_proj = nn.Sequential(nn.Linear(768, self.hidden_size))
        self.set_if_finetune(self.audio_mae_proj, config.model.finetune.audio_mae_proj)

        self.image_mae = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.set_if_finetune(self.image_mae, config.model.finetune.image_mae)

        self.image_mae_proj = nn.Sequential(nn.Linear(1024, self.hidden_size),)
        self.set_if_finetune(self.image_mae_proj, config.model.finetune.image_mae_proj)
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # CLIP-Align
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.dropout = nn.Dropout(self.config.model.dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        self.clip_loss_fn = ClipLoss()
        self.cosine_loss_fn = nn.CosineEmbeddingLoss()
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.6592), requires_grad=True)

    def forward(self, unbg_text_input_ids, unbg_text_attention_mask, audio_feats, frame_feats, labels):
        audio_feats = self.audio_mae.forward_encoder_no_mask(audio_feats)
        audio_feats = self.audio_mae_proj(audio_feats)
        audio_feats_mean = torch.mean(audio_feats, dim=1)
        audio_feats_mean = audio_feats_mean / audio_feats_mean.norm(dim=-1, keepdim=True)

        frame_feats = self.image_mae(frame_feats).last_hidden_state
        frame_feats = self.image_mae_proj(frame_feats)
        frame_feats_mean = torch.mean(frame_feats, dim=1)
        frame_feats_mean = frame_feats_mean / frame_feats_mean.norm(dim=-1, keepdim=True)

        # GNN fusion
        fusion_feats = self.mmvig(image_embeds=frame_feats, audio_embeds=audio_feats)
        # fusion_feats = self.fusion_linear(fusion_feats)
        fusion_feats = fusion_feats / fusion_feats.norm(dim=-1, keepdim=True)

        # fusion feat inter-class align
        selected_labels = self.sample_unique_labels(labels)
        selected_fusion_feats = torch.stack([frame_feats[i] for i in selected_labels], dim=0)
        selected_fusion_feats_2 = torch.cat([selected_fusion_feats[1:], selected_fusion_feats[:1]], dim=0)
        inter_cosine_loss = self.cosine_loss_fn(selected_fusion_feats, selected_fusion_feats_2, torch.tensor([-1] * selected_fusion_feats.shape[0]).to(selected_fusion_feats.device))

        # fusion feat inter-class align
        selected_labels = self.sample_unique_labels(labels)
        selected_fusion_feats = torch.stack([fusion_feats[i] for i in selected_labels], dim=0)
        selected_fusion_feats_2 = torch.cat([selected_fusion_feats[1:], selected_fusion_feats[:1]], dim=0)
        cosine_loss = self.cosine_loss_fn(selected_fusion_feats, selected_fusion_feats_2, torch.tensor([-1] * selected_fusion_feats.shape[0]).to(selected_fusion_feats.device))
        # cosine_loss = torch.tensor(0.0).to(cls_loss.device)

        fusion_feats = torch.cat([fusion_feats, audio_feats_mean, frame_feats_mean], dim=1)
        fusion_feats = self.dropout(fusion_feats)
        logits = self.classifier(fusion_feats)
        cls_loss = self.loss_fn(logits, labels)

        av_similarity_labels = torch.tensor([-1.0 if labels[i] == 0 else 1.0 for i in range(len(labels))]).to(cls_loss.device)
        av_cosine_loss = self.cosine_loss_fn(audio_feats_mean, frame_feats_mean, av_similarity_labels) + inter_cosine_loss

        total_loss = cls_loss + av_cosine_loss + cosine_loss
        return total_loss, cls_loss, av_cosine_loss, cosine_loss, logits

    def predict(self, unbg_text_input_ids, unbg_text_attention_mask, audio_feats, frame_feats, labels):
        _, _, _, _, logits = self.forward(unbg_text_input_ids, unbg_text_attention_mask, audio_feats, frame_feats, labels)
        logits = torch.softmax(logits, dim=1)
        return logits

    def set_if_finetune(self, model, if_finetune):
        for param in model.parameters():
            param.requires_grad = if_finetune

    def sample_unique_labels(self, labels):
        np_labels = labels.cpu().numpy()
        # np_labels = [label for label in np_labels if label != 0]
        unique_labels = set(np_labels)
        indices = []
        for label in unique_labels:
            indice = np.where(np_labels == label)[0][0]
            indices.append(indice)
        indices = shuffle(indices)
        return indices#[:8]
