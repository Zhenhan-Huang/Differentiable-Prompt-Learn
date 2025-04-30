import sys
import os.path as osp
import time
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.data import DataManager
from dassl.data.datasets import build_dataset
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import build_transform, INTERPOLATION_MODES
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import (
    load_pretrained_weights, load_checkpoint, mkdir_if_missing,
    MetricMeter, AverageMeter
)
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from tqdm import tqdm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES, CUSTOM_TEMPLATES

_tokenizer = _Tokenizer()

 
def load_clip_to_cpu(cfg, design_details):
    backbone_name = design_details['backbone']
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, alphas_txt, ctx_txt_lst):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, alphas_txt, ctx_txt_lst, 0]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape: [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        assert torch.cuda.is_available()
        n_cls = len(classnames)
        ctx_init = cfg.TRAINER.TRAINERSUBNET.CTX_INIT
        alpha_path_img = cfg.TRAINER.TRAINERSUBNET.IMG_ALPHA_PATH
        alpha_path_txt = cfg.TRAINER.TRAINERSUBNET.TXT_ALPHA_PATH
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        classnames = [name.replace("_", " ") for name in classnames]
        if ctx_init == 'ensemble':
            embedding_all = []
            with torch.no_grad():
                for single_template in IMAGENET_TEMPLATES:
                    prompts = [single_template.replace("{}", name) for name in classnames]      
                    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
                    embed_single = clip_model.token_embedding(tokenized_prompts).type(dtype)
                    embedding_all.append(embed_single.unsqueeze(1))
            self.embedding = torch.cat(embedding_all, dim=1).mean(dim=1)

        elif ctx_init == 'custom':
            ctx_template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            prompts = [ctx_template.replace("{}", name) for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.embedding = embedding

        elif ctx_init == 'default':
            ctx_pre = 'a photo of a'
            prompts = [ctx_pre + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.embedding = embedding

        else:
            raise ValueError(f'Initial context option "{ctx_init}" not supported')

        alpha_img = torch.from_numpy(np.load(alpha_path_img))
        edge_img = F.softmax(alpha_img, dim=-1)
        edge_idx_img = torch.argmax(edge_img, dim=-1)
        ctx_img_lst = []
        for idx in edge_idx_img:
            ctx_img_lst.append(torch.empty(idx, 768))
        self.ctx_img_lst = nn.ParameterList(ctx_img_lst)
        for param in self.ctx_img_lst:
            nn.init.normal_(param, std=0.02)

        alpha_txt = torch.from_numpy(np.load(alpha_path_txt))
        edge_txt = F.softmax(alpha_txt, dim=-1)
        edge_idx_txt = torch.argmax(edge_txt, dim=-1)
        ctx_txt_lst = []
        for idx in edge_idx_txt:
            ctx_txt_lst.append(torch.empty(idx, ctx_dim))
        self.ctx_txt_lst = nn.ParameterList(ctx_txt_lst)
        for param in self.ctx_txt_lst:
            nn.init.normal_(param, std=0.02)

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts

        tea_details = {
            "trainer": "IVLP", "backbone": cfg.MODEL.BACKBONE.NAME,
            "vision_depth": 0, "vision_ctx": 0,
            "language_depth": 0, "language_depth": 0
        }
        clip_zs = load_clip_to_cpu(cfg, tea_details).float().cuda()
        self.ZS_image = clip_zs.visual
        with torch.no_grad():
            all_teacher_features = []
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_zs.encode_text(x_tokenized.cuda())
            all_teacher_features.append(text_features.unsqueeze(1))
        self.zs_text_ft = torch.cat(all_teacher_features, dim=1).mean(dim=1)

    def arch_parameters(self):
        return self.alphas

    def forward(self):
        return self.embedding.cuda(), self.ctx_img_lst, self.ctx_txt_lst
    

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, task=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        embedding, ctx_img_lst, ctx_txt_lst = self.prompt_learner()
        # (num_classes, hidden_txt_dimension)
        text_embed = self.text_encoder(embedding, tokenized_prompts, None, ctx_txt_lst)
        # (batch_size, PhxPw+1, hidden_txt_dimension)
        image_embed = self.image_encoder(image.type(self.dtype), None, ctx_img_lst)

        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_embed @ text_embed.t()

        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            zs_text_embed = self.prompt_learner.zs_text_ft  # precomputed pre-trained frozen textual features
            zs_text_embed = zs_text_embed / zs_text_embed.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zs_image_embed = self.prompt_learner.ZS_image(image.type(self.dtype))
                zs_image_embed = zs_image_embed / zs_image_embed.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zs_image_embed.cuda() @ zs_text_embed.cuda().t()

            return zero_shot_logits, logits

        return logits


@TRAINER_REGISTRY.register()
class TrainerSubnet(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.TRAINERSUBNET.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        design_details = {
            "trainer": "TrainerSubnet",
            "backbone": cfg.MODEL.BACKBONE.NAME,
        }
        clip_model = load_clip_to_cpu(cfg, design_details)

        if (cfg.TRAINER.TRAINERSUBNET.PREC == "fp32" or
            cfg.TRAINER.TRAINERSUBNET.PREC == "amp"):
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                # freeze param in prompt learner
                if "ZS_image" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.TRAINERSUBNET.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            raise RuntimeError('MultiGPU significantly slows down the program, exit the program now.')

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.TRAINERSUBNET.PREC
        if prec == 'amp':
            raise NotImplementedError
        else:
            zero_shot_logits, logits= model(image, label)
            loss_div = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            loss_ce = F.cross_entropy(logits, label)
            loss = loss_div*0 + loss_ce
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        loss_summary = {
            'loss_train': loss.item(),
            'loss_ce': loss_ce.item(),
            'loss_div': loss_div.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary 
 
    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                print("Delete 'token_prefix' of 'prompt_learner'")
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                print("Delete 'token_suffix' of 'prompt_learner'")
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            msg = self._models[name].load_state_dict(state_dict, strict=False)
            print(f'Load pretrain: {msg}')

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        loss_sum = 0.
        count_sum = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            # self.model_inference(input) calls self.model(input)
            output = self.model_inference(input)
            loss = F.cross_entropy(output, label)
            loss_sum += loss * input.shape[0]
            count_sum += input.shape[0]
            self.evaluator.process(output, label)

        print(f'Average test loss is {loss_sum/count_sum:.2f}')

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def return_test(self):
        if self.cfg.TEST.FINAL_MODEL == "best_val":
            print("Deploy the model with the best val performance")
            self.load_model(self.output_dir)
        else:
            print("Deploy the last-epoch model")
        
        acc = self.test()

        return acc

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME

        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        if self.cfg.USE_WRITER:
            writer_dir = osp.join(self.output_dir, "tensorboard")
            mkdir_if_missing(writer_dir)
            self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()
