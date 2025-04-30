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
from .imagenet_templates import IMAGENET_TEMPLATES
from torch.autograd import Variable

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
        ctx_init = cfg.TRAINER.TRAINERSUPERNET.CTX_INIT
        ctx_img_depth = 12
        nctx_img = cfg.TRAINER.TRAINERSUPERNET.N_CTX_VISION
        ctx_txt_depth = 12
        nctx_txt = cfg.TRAINER.TRAINERSUPERNET.N_CTX_TEXT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_init = ctx_init.replace("_", " ")
        
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [ctx_init + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.embedding = embedding

        self.alphas_img = Variable(1.0e-3*torch.randn(
            ctx_img_depth, 1+nctx_img, dtype=dtype).cuda(), requires_grad=True)

        ctx_img_lst = []
        for _ in range(ctx_img_depth):
            ctx_img_lst.append(
                nn.ParameterList([
                    nn.Parameter(torch.randn(c+1, 768, dtype=dtype)) for c in range(nctx_img)
                ])
            )
        self.ctx_img_lst = nn.ParameterList(ctx_img_lst)

        self.alphas_txt = Variable(1.0e-3*torch.randn(
            ctx_txt_depth, 1+nctx_txt, dtype=dtype).cuda(), requires_grad=True)

        ctx_txt_lst = []
        for _ in range(ctx_txt_depth):
            ctx_txt_lst.append(
                nn.ParameterList([
                    nn.Parameter(torch.rand(c+1, ctx_dim, dtype=dtype)) for c in range(nctx_txt)
                ])
            )
        self.ctx_txt_lst = nn.ParameterList(ctx_txt_lst)

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts

        # tea_details = {
        #     "trainer": "IVLP", "backbone": cfg.MODEL.BACKBONE.NAME,
        #     "vision_depth": 0, "vision_ctx": 0,
        #     "language_depth": 0, "language_depth": 0
        # }
        # clip_zs = load_clip_to_cpu(cfg, tea_details).float().cuda()
        # self.ZS_image = clip_zs.visual
        # with torch.no_grad():
        #     all_teacher_features = []
        #     for single_template in IMAGENET_TEMPLATES:
        #         x = [single_template.replace("{}", name) for name in classnames]
        #         x_tokenized = torch.cat([clip.tokenize(p) for p in x])
        #         text_features = clip_zs.encode_text(x_tokenized.cuda())
        #     all_teacher_features.append(text_features.unsqueeze(1))
        # self.zs_text_ft = torch.mean(all_teacher_features, dim=1)

        self.alphas = [self.alphas_img, self.alphas_txt]

    def arch_parameters(self):
        return self.alphas

    def forward(self):
        return self.embedding.cuda(), self.alphas_img, self.alphas_txt, \
            self.ctx_img_lst, self.ctx_txt_lst
    

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

        embedding, alphas_img, alphas_txt, ctx_img_lst, ctx_txt_lst = self.prompt_learner()
        # (num_classes, hidden_txt_dimension)
        text_embed = self.text_encoder(embedding, tokenized_prompts, alphas_txt, ctx_txt_lst)
        # (batch_size, PhxPw+1, hidden_txt_dimension)
        image_embed = self.image_encoder(image.type(self.dtype), alphas_img, ctx_img_lst)

        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_embed @ text_embed.t()

        return logits


@TRAINER_REGISTRY.register()
class TrainerSupernet(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.TRAINERSUPERNET.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        design_details = {
            "trainer": "TrainerSupernet",
            "backbone": cfg.MODEL.BACKBONE.NAME,
        }
        clip_model = load_clip_to_cpu(cfg, design_details)

        if (cfg.TRAINER.TRAINERSUPERNET.PREC == "fp32" or
            cfg.TRAINER.TRAINERSUPERNET.PREC == "amp"):
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

        self.optimizer = torch.optim.Adam(
            self.model.prompt_learner.arch_parameters(),
            lr=0.0035,
            betas=(0.5, 0.999),
            weight_decay=0
        )

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        if cfg.TRAINER.TRAINERSUPERNET.PREC == "amp":
            self.scaler_arch = GradScaler()
            self.scaler_net = GradScaler()
        else:
            self.scaler_arch = None
            self.scaler_net = None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            raise RuntimeError('MultiGPU significantly slows down the program, exit the program now.')

        # save alphas before training 
        self.epoch = -1
        self.save_alpha(self.model.prompt_learner.arch_parameters())

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        assert self.val_loader is not None, (
            "Validation loader is None. It is required to optimize alphas"
        )

        end = time.time()
        for self.batch_idx, batch_train in enumerate(self.train_loader_x):
            batch_val = next(iter(self.val_loader))
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_train, batch_val)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        self.save_alpha(self.model.prompt_learner.arch_parameters())

    def save_alpha(self, alpha_tuple):
        alpha_img = alpha_tuple[0]
        alpha_txt = alpha_tuple[1]
        txt_name = f'{self.cfg.OUTPUT_DIR}/alpha_e{self.epoch+1}_txt.npy'
        np.save(txt_name, alpha_img.cpu().detach().numpy())
        img_name = f'{self.cfg.OUTPUT_DIR}/alpha_e{self.epoch+1}_img.npy'
        np.save(img_name, alpha_txt.cpu().detach().numpy())

    def forward_backward_alphas(
        self, image_val, label_val, task_val, optimizer, model, prec):
        scaler = self.scaler_arch
        if prec == 'amp':
            with autocast():
                logits = model(image_val, task_val)
                loss = F.cross_entropy(logits, label_val)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(image_val, task_val)
            loss = F.cross_entropy(logits, label_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_summary = {'loss_val': loss.item()}
        return loss_summary

    def forward_backward_model(
        self, image_train, label_train, task_train, optim, model, prec):
        scaler = self.scaler_net
        if prec == 'amp':
            with autocast():
                logits = model(image_train, task_train)
                loss = F.cross_entropy(logits, label_train)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(image_train, task_train)
            loss = F.cross_entropy(logits, label_train)
            optim.zero_grad()
            loss.backward()
            optim.step()
        loss_summary = {'loss_train': loss.item()}
        return loss_summary 

    def forward_backward(self, batch_train, batch_val):
        prec = self.cfg.TRAINER.TRAINERSUPERNET.PREC
        image_train, label_train, task_train = self.parse_batch_train(batch_train)
        image_val, label_val, task_val = self.parse_batch_test(batch_val)
        model = self.model
        optim = self.optim
        optimizer = self.optimizer
        loss_summary = {}

        loss_tmp = self.forward_backward_alphas(
            image_val, label_val, task_val, optimizer, model, prec)
        loss_summary.update(loss_tmp)
        loss_tmp = self.forward_backward_model(
            image_train, label_train, task_train, optim, model, prec)
        loss_summary.update(loss_tmp)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
 
    def parse_batch_train(self, batch):
        if self.cfg.DATASET.DATATYPE == 'coop':
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            raise NotImplementedError
        input = batch[inp_key]
        label = batch[lab_key]
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, tasks

    def parse_batch_test(self, batch):
        if self.cfg.DATASET.DATATYPE == 'coop':
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            raise NotImplementedError
        input = batch[inp_key]
        label = batch[lab_key]
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, tasks

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
            input, label, task = self.parse_batch_test(batch)
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

    def build_data_loader(self):
        self.multi_task = self.cfg.DATASET.MULTITASK

        if self.cfg.DATASET.DATATYPE == "coop":
            dm = MCoopDataManager(self.cfg)
        else:
            raise ValueError(f"Dataset type '{self.cfg.DATASET.DATATYPE}' not supported")
        
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm


class MCoopDataManager(DataManager):

    def __init__(
        self, cfg, custom_tfm_train=None, custom_tfm_test=None,
        dataset_wrapper=None
    ):
        label_offset = 0
        self.num_classes_list = []
        self.classnames_list = []
        self.lab2cname_list = {}
        self.dataset = None
        self._task_names = cfg.DATASET.DATASET.split(',')
        self._id2task = {}
        self._task_class_idx = {}

        for domain, dataset_name in enumerate(self._task_names):
            cfg.defrost()
            cfg.DATASET.NAME = dataset_name
            cfg.freeze()
            self._id2task[domain] = dataset_name
            dataset = build_dataset(cfg)
            self.num_classes_list.append(dataset._num_classes)
            self.classnames_list += dataset._classnames
            new_lab2cname_dict = {}
            for key, value in dataset._lab2cname.items():
                new_lab2cname_dict[key+label_offset] = value
            self.lab2cname_list.update(new_lab2cname_dict)
            for i in range(len(dataset._train_x)):
                dataset._train_x[i]._label += label_offset
                dataset._train_x[i]._domain = domain
            
            if dataset._train_u:
                for i in range(len(dataset._train_u)):
                    dataset._train_u[i]._label += label_offset
                    dataset._train_u[i]._domain = domain
                if self.dataset is not None:
                    self.dataset._train_u = self.dataset._train_u + dataset._train_u
            if dataset._val:
                for i in range(len(dataset._val)):
                    dataset._val[i]._label += label_offset
                    dataset._val[i]._domain = domain

            for i in range(len(dataset._test)):
                dataset._test[i]._label += label_offset
                dataset._test[i]._domain = domain
            
            if self.dataset is not None:
                self.dataset._train_x = self.dataset._train_x + dataset._train_x
                self.dataset._val = self.dataset.val + dataset.val
                self.dataset._test = self.dataset.test + dataset.test

            print(
                f'TrainU is None: {dataset._train_u is None}, '
                f'TrainVal is None: {dataset._val is None}'
            )
            if self.dataset is None:
                self.dataset = dataset

            self._task_class_idx[dataset_name] = (
                label_offset, label_offset + dataset._num_classes)
            label_offset += dataset._num_classes
            
        dataset = self.dataset
        dataset._classnames = self.classnames_list
        dataset._lab2cname = self.lab2cname_list
        dataset._num_classes = sum(self.num_classes_list)
        print(
            f'Number of classes in each dataset: {self.num_classes_list}.\n'
            f'Total number of class names is {len(dataset._classnames)}.\n'
            f'Label ID to class name: {dataset._lab2cname}.\n'
            f'Total number of classes is {dataset._num_classes}.'
        )
           
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)