from hash_model import HashingModel, VisualTextualPrompting, MITH
import time

from einops import rearrange
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
import torch.nn.functional as F

from model.clip_model.model import load_download_clip
from optimization import BertAdam
from utils.calc_utils import calc_neighbor, calc_map_k, calc_map_k_ind, calc_hamming_dist
from load_data import generate_dataset
from utils import get_logger
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import shutil
from model.clip_model.simple_tokenizer import SimpleTokenizer


dataset_root_path = "./dataset/"
main_path = './'


class TrainerAsym:
    def __init__(self, args):

        self.args = args

        torch.random.manual_seed(seed=self.args.seed)
        torch.autograd.set_detect_anomaly(True)
        # torch.backends.cudnn.enabled = False

        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu

        os.makedirs(self.args.save_dir, exist_ok=True)
        self._init_writer()

        self.logger.info('Start logging...')

        if self.args.is_train:
            log_str_args = "\n"
            for para in self.args.__dict__:
                log_str_args += " " * (30 - len(para)) + str(para) + "=" + str(self.args.__dict__[para]) + "\n"
            self.logger.info(log_str_args)
        else:
            self.logger.info(f"pretrained: {self.args.pretrained}")

        self.rank = self.args.rank  # gpu rank

        self._init_dataset()
        self._init_model()

        self.dim = 512
        self.patch_num = 49
        self.max_mapi2t = 0
        self.best_epoch_i2t = 0
        self.max_mapt2i = 0
        self.best_epoch_t2i = 0
        self.max_map = {'i2t': 0, "t2i": 0}
        self.best_epoch = 0

        self.logger.info("Train dataset len: {}".format(len(self.train_loader.dataset)))
        self.k_bits = self.args.k_bits

        self.img_buffer_tokens = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)
        self.img_buffer_cls = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)

        self.txt_buffer_tokens = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)
        self.txt_buffer_cls = torch.randn(self.args.train_num, self.k_bits).to(self.rank, non_blocking=True)

        self.device = self.txt_buffer_cls.device
        self.run()

    def run(self):
        if self.args.is_train:
            self.train()
        else:
            self.test()

    def _init_writer(self):
        self.logger = get_logger(os.path.join(self.args.save_dir, "train.log" if self.args.is_train else "test.log"))

    def _init_model(self):
        self.logger.info("init model.")
        self.logger.info("Using ViT & GPT2...")

        self.clip, clip_info = load_download_clip(self.args.clip_path)
        self.vtp = VisualTextualPrompting(args=self.args).to(self.rank)
        self.mith_hash = HashingModel(clip_info=clip_info, args=self.args).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info(f"load pretrained model at {self.args.pretrained}")
            self.load_model(self.args.pretrained)

        self.clip = self.clip.to(self.rank)
        self.clip.float()
        self.clip.eval()
        self.clip.requires_grad_(False)
        self.vtp.float()
        self.mith_hash.float()

        self.optimizer = BertAdam(
            [{'params': self.mith_hash.parameters(), 'lr': self.args.lr},
             {'params': self.vtp.parameters(), 'lr': self.args.lr}],
            lr=self.args.lr,
            warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset...")

        global dataset_root_path

        self.args.index_file = os.path.join(dataset_root_path, self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join(dataset_root_path, self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join(dataset_root_path, self.args.dataset, self.args.label_file)
        self.args.img_caption_file = os.path.join(dataset_root_path, self.args.dataset, self.args.img_caption_file)

        train_data, query_data, retrieval_data = generate_dataset(captionFile=self.args.caption_file,
                                                                  indexFile=self.args.index_file,
                                                                  labelFile=self.args.label_file,
                                                                  img_captionFile=self.args.img_caption_file,
                                                                  maxWords=self.args.max_words,
                                                                  imageResolution=self.args.resolution,
                                                                  query_num=self.args.query_num,
                                                                  train_num=self.args.train_num,
                                                                  seed=self.args.seed,
                                                                  noise_ratio=self.args.noise_ratio,
                                                                  dataset=self.args.dataset)

        self.train_labels = train_data.get_all_label().float()
        self.query_labels = query_data.get_all_label().float()
        self.retrieval_labels = retrieval_data.get_all_label().float()

        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=False
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=False
        )

        self.query_data = query_data
        self.retrieval_data = retrieval_data

    def change_state(self, mode):
        if mode == "train":
            self.vtp.train()
            self.mith_hash.train()
        elif mode == "valid":
            self.vtp.eval()
            self.mith_hash.eval()

    def save_model(self):
        torch.save({"vtp": self.vtp.state_dict(), "hash": self.mith_hash.state_dict()}, os.path.join(self.args.save_dir, "model.pth"))

    def load_model(self, model_path,):
        check_points = torch.load(model_path, map_location=f"cuda:{self.rank}")
        self.vtp.load_state_dict(check_points["vtp"])
        self.mith_hash.load_state_dict(check_points["hash"])

    def train_epoch(self, epoch):

        self.logger.info("\n\n\n")
        self.logger.info(
            "####################### Train epochs: %d/%d #######################" % (epoch, self.args.epochs))
        epoch_avg_loss_dict = {'all_loss': 0}

        self.change_state(mode="train")

        for image, text, key_padding_mask, label, index, img_caption, key_padding_mask_cap, _ in self.train_loader:

            image = image.float().to(self.rank, non_blocking=True)
            img_caption = img_caption.to(self.rank, non_blocking=True)
            key_padding_mask_cap = key_padding_mask_cap.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)
            label = label.float().to(self.rank, non_blocking=True)

            img_tokens, _, img_cls = self.clip.encode_image(image)
            txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)
            cap_tokens, _, new_key_padding_mask_cap, cap_eos = self.clip.encode_text(img_caption, key_padding_mask_cap)

            img_tokens_vtp, img_cls_vtp, txt_tokens_vtp, txt_eos_vtp, txt_eos_prompted = self.vtp(img_tokens, img_cls, txt_tokens, txt_eos, cap_tokens)

            output_dict = self.mith_hash(img_tokens_vtp, txt_tokens_vtp, img_cls_vtp, txt_eos_vtp, new_key_padding_mask)

            output_dict["txt_eos"] = txt_eos
            output_dict["cap_eos"] = cap_eos
            output_dict["txt_eos_prompted"] = txt_eos_prompted

            self.img_buffer_cls[index] = output_dict['img_cls_hash'].detach()
            self.txt_buffer_cls[index] = output_dict['txt_cls_hash'].detach()
            self.img_buffer_tokens[index] = output_dict['img_tokens_hash'].detach()
            self.txt_buffer_tokens[index] = output_dict['txt_tokens_hash'].detach()

            hyper_lambda = self.args.hyper_lambda
            B = torch.sign(
                (output_dict['img_cls_hash'].detach() * hyper_lambda + output_dict['img_tokens_hash'].detach() * (1 - hyper_lambda)) + \
                (output_dict['txt_cls_hash'].detach() * hyper_lambda + output_dict['txt_tokens_hash'].detach() * (1 - hyper_lambda)))

            ALL_LOSS_DICT = self.compute_loss(output_dict, label, B)

            loss = 0
            for key in ALL_LOSS_DICT:
                loss += ALL_LOSS_DICT[key]
                if key in epoch_avg_loss_dict:
                    epoch_avg_loss_dict[key] += ALL_LOSS_DICT[key]
                else:
                    epoch_avg_loss_dict[key] = ALL_LOSS_DICT[key]

            epoch_avg_loss_dict['all_loss'] += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] all loss avg: {epoch_avg_loss_dict['all_loss'].data / (len(self.train_loader))}")
        self.logger.info(f"lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def train(self):
        self.logger.info("Start train...")

        for epoch in range(self.args.epochs):
            time1 = time.time()
            self.train_epoch(epoch)

            time2 = time.time()
            spend_time = int(time2 - time1)
            self.logger.info(
                f"{self.args.dataset}_{self.args.k_bits}. Train epoch [{epoch}], spend {spend_time // 60} min, {spend_time % 60} sec")

            if (epoch + 1) % self.args.valid_freq == 0:
                self.valid(epoch)

            time3 = time.time()
            spend_time = int(time3 - time2)
            self.logger.info(
                f"{self.args.dataset}_{self.args.k_bits}. Valid epoch [{epoch}], spend {spend_time // 60} min, {spend_time % 60} sec")
            self.logger.info(
                f"I-T Best epoch: {self.best_epoch_i2t}, mAP: {self.max_mapi2t}, T-I Best epoch: {self.best_epoch_t2i}, mAP: {self.max_mapt2i}")

        self.logger.info(f">>>>>>> FINISHED {self.args.dataset}_{self.args.k_bits}. <<<<<<<")
        self.logger.info(f"Best epoch: {self.best_epoch}, i2t = {self.max_map['i2t']} t2i = {self.max_map['t2i']}")
        self.logger.info(
            f"I-T Best epoch: {self.best_epoch_i2t}, mAP: {self.max_mapi2t}, T-I Best epoch: {self.best_epoch_t2i}, mAP: {self.max_mapt2i}")

    def valid(self, epoch):
        self.logger.info("\n")
        self.logger.info(" Valid: %d/%d " % (epoch, self.args.epochs))
        self.change_state(mode="valid")
        # TODO
        test_caption_path = os.path.join(main_path, 'path/to/your/test/image_caption')
        test_image_dir = os.path.join(main_path, 'path/to/your/test/generated_image')

        q_t = self.get_text_code_LMA(test_caption_path, test_image_dir)
        q_i, _ = self.get_code(self.query_loader, self.args.query_num, split="query")
        r_i, r_t = self.get_code(self.retrieval_loader, self.args.retrieval_num, split="retrieval")

        _k_ = None
        mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), self.query_labels.to(self.device),
                            self.retrieval_labels.to(self.device), _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), self.query_labels.to(self.device),
                            self.retrieval_labels.to(self.device), _k_).item()

        if mAPi2t + mAPt2i > self.max_map['i2t'] + self.max_map['t2i']:
            self.best_epoch = epoch
            self.max_map['i2t'] = mAPi2t
            self.max_map['t2i'] = mAPt2i
            self.logger.info("$$$$$$$$$$$$$$$$$$$$ Best maps. $$$$$$$$$$$$$$$$$$$$$$$$")
            self.save_model()
            self.logger.info(
                f"{self.args.dataset}_{self.args.k_bits}. Epoch: {epoch}, save model to {os.path.join(self.args.save_dir, 'model.pth')}")
            self.save_mat(q_i, q_t, r_i, r_t, epoch)

        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i2t = epoch
            self.max_mapi2t = max(self.max_mapi2t, mAPi2t)

        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t2i = epoch
            self.max_mapt2i = max(self.max_mapt2i, mAPt2i)

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}]")
        self.logger.info(f"MAP(i->t): {round(mAPi2t, 5)}, MAP(t->i): {round(mAPt2i, 5)}")
        self.logger.info(
            f"MAX [{self.best_epoch}/{self.args.epochs}] MAP(i->t): {round(self.max_map['i2t'], 5)}, MAP(t->i): {round(self.max_map['t2i'], 5)}")

    def get_code(self, data_loader, length: int, split='retrieval'):
        k_bits = self.k_bits

        img_buffer = torch.empty(length, k_bits, dtype=torch.float).to(self.device)
        text_buffer = torch.empty(length, k_bits, dtype=torch.float).to(self.device)

        # txt_cls_buffer = torch.empty(length, 512, dtype=torch.float).to(self.device)

        for image, text, key_padding_mask, label, index, img_caption, key_padding_mask_cap, _ in tqdm(data_loader):
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)
            index = index.numpy()
            img_caption = img_caption.to(self.rank, non_blocking=True)
            key_padding_mask_cap = key_padding_mask_cap.to(self.rank, non_blocking=True)
            img_caption = img_caption.to(self.rank, non_blocking=True)
            key_padding_mask_cap = key_padding_mask_cap.to(self.rank, non_blocking=True)

            img_tokens, _, img_cls = self.clip.encode_image(image)
            txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)
            cap_tokens, _, new_key_padding_mask_cap, cap_eos = self.clip.encode_text(img_caption, key_padding_mask_cap)
            # txt_cls_buffer[index] = txt_eos
            if split == 'retrieval':
                img_tokens, img_cls, txt_tokens, txt_eos, txt_eos_prompted = self.vtp(img_tokens, img_cls, txt_tokens, txt_eos, cap_tokens)
            elif split == 'query':
                img_tokens, img_cls = self.vtp.forward_image(img_tokens, img_cls)
                txt_tokens, txt_eos = self.vtp.forward_text(txt_tokens, txt_eos)
            else:
                raise Exception("Split Mode Error.")
            output_dict = self.mith_hash(img_tokens, txt_tokens, img_cls, txt_eos, new_key_padding_mask)

            img_hash_cls = output_dict['img_cls_hash'].detach()
            txt_hash_cls = output_dict['txt_cls_hash'].detach()

            img_tokens_hash = output_dict['img_tokens_hash'].detach()
            txt_tokens_hash = output_dict['txt_tokens_hash'].detach()

            img_buffer[index, :] = torch.sign(img_tokens_hash + img_hash_cls)
            text_buffer[index, :] = torch.sign(txt_tokens_hash + txt_hash_cls)

        return img_buffer, text_buffer

    def get_text_code_LMA(self, test_caption_path, test_image_dir):
        test_capions = []
        with open(test_caption_path, 'r') as f:
            for line in f:
                test_capions.append(line)
        test_capions = np.array(test_capions)

        q_t = torch.empty(self.args.query_num, self.args.k_bits, dtype=torch.float).to(self.device)

        for _, text, key_padding_mask, _, index, _, _, _ in tqdm(self.query_loader):
            index = index.numpy()
            image = self.load_batch_images(index, test_image_dir).to(self.rank, non_blocking=True)
            img_caption, key_padding_mask_cap = self.load_texts(test_capions[index])

            text = text.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)
            img_caption = img_caption.to(self.rank, non_blocking=True)
            key_padding_mask_cap = key_padding_mask_cap.to(self.rank, non_blocking=True)

            img_tokens, _, img_cls = self.clip.encode_image(image)
            txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)
            cap_tokens, _, new_key_padding_mask_cap, cap_eos = self.clip.encode_text(img_caption, key_padding_mask_cap)
            # txt_cls_buffer[index] = txt_eos
            img_tokens, img_cls, txt_tokens, txt_eos, txt_eos_prompted = self.vtp(img_tokens, img_cls, txt_tokens,
                                                                                  txt_eos, cap_tokens)

            output_dict = self.mith_hash(img_tokens, txt_tokens, img_cls, txt_eos, new_key_padding_mask)

            txt_hash_cls = output_dict['txt_cls_hash'].detach()
            txt_tokens_hash = output_dict['txt_tokens_hash'].detach()

            q_t[index, :] = torch.sign(txt_tokens_hash + txt_hash_cls)
        return q_t

    def load_batch_images(self, index, root_path):
        transform = Compose([
            Resize((224, 224), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        images = []
        for i in index:
            image_path = os.path.join(root_path, str(int(i)+1)+'.png')
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image)
            images.append(image_tensor.unsqueeze(0))

        return torch.cat(images, dim=0)

    def load_texts(self, texts):
        maxWords = 32
        tokenizer = SimpleTokenizer()
        SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                         "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        captions = []
        key_padding_masks = []
        for i in texts:
            words = tokenizer.tokenize(i)
            words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = maxWords - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
            caption = tokenizer.convert_tokens_to_ids(words)
            while len(caption) < maxWords:
                caption.append(0)
            caption = torch.tensor(caption)
            key_padding_mask = (caption == 0)

            captions.append(caption.unsqueeze(0))
            key_padding_masks.append(key_padding_mask.unsqueeze(0))

        return torch.cat(captions, dim=0), torch.cat(key_padding_masks, dim=0)

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, epoch):

        save_dir = os.path.join(self.args.save_dir, "PR_curve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.cpu().detach().numpy()
        retrieval_labels = self.retrieval_labels.cpu().detach().numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }

        scio.savemat(
            os.path.join(save_dir,
                         f"VTPH_" + self.args.dataset + "_" + str(self.args.k_bits) + "_epoch" + str(epoch) + ".mat"),
            result_dict)
        self.logger.info(f">>>>>> save best *.mat data!")

    def compute_loss(self, output_dict, label, B):
        ALL_LOSS = {}

        label_sim = calc_neighbor(self.train_labels.float().to(self.rank, non_blocking=True), label)

        img_tokens_hash = output_dict['img_tokens_hash']
        txt_tokens_hash = output_dict['txt_tokens_hash']

        img_cls_hash = output_dict['img_cls_hash']
        txt_cls_hash = output_dict['txt_cls_hash']

        res_img_cls = output_dict['res_img_cls']
        res_txt_eos = output_dict['res_txt_cls']

        # Token Intra
        # hyper_tokens_intra = self.args.hyper_tokens_intra
        ALL_LOSS['tokens_intra_likelihood'] = self.bayesian_loss(self.img_buffer_tokens, img_tokens_hash,
                                                                 label_sim) + self.bayesian_loss(self.txt_buffer_tokens,
                                                                                                 txt_tokens_hash,
                                                                                                 label_sim)

        # CLS Inter
        hyper_cls_inter = self.args.hyper_alpha
        ALL_LOSS['cls_inter_likelihood'] = hyper_cls_inter * \
                                           (self.bayesian_loss(self.img_buffer_cls, txt_cls_hash, label_sim) +
                                            self.bayesian_loss(self.txt_buffer_cls, img_cls_hash, label_sim))

        # hash feature
        H_i = img_cls_hash * 0.5 + img_tokens_hash * 0.5
        H_t = txt_cls_hash * 0.5 + txt_tokens_hash * 0.5
        # quantization loss
        hyper_quan = self.args.hyper_beta
        ALL_LOSS['quantization'] = hyper_quan * (self.quantization_loss_2(H_i, B) + self.quantization_loss_2(H_t, B))

        feature_diff_it = self.cal_feature_differ(output_dict['cap_eos'], output_dict['txt_eos'])
        feature_diff_ti = self.cal_feature_differ(output_dict['txt_eos'], output_dict['cap_eos'])
        f_diff = feature_diff_it + feature_diff_ti
        hyper_gamma = self.args.hyper_gamma
        temperature = 0.07 + hyper_gamma * f_diff

        # Contrastive Alignment loss
        hyper_info_nce = self.args.hyper_zeta

        ALL_LOSS['infoNCE'] = hyper_info_nce * self.info_nce_loss(res_img_cls, res_txt_eos, temperature=temperature)

        # 1*gradient back to student.
        item1 = (F.mse_loss(img_cls_hash.detach(), img_tokens_hash, reduction='sum') +
                 F.mse_loss(txt_cls_hash.detach(), txt_tokens_hash, reduction='sum'))
        # 0.1*gradient back to teacher.
        item2 = 0.1 * (F.mse_loss(img_cls_hash, img_tokens_hash.detach(), reduction='sum') +
                       F.mse_loss(txt_cls_hash, txt_tokens_hash.detach(), reduction='sum'))

        # distillation loss
        ALL_LOSS['distillation'] = (item1 + item2) / (img_cls_hash.shape[0])

        ALL_LOSS['triplet'] = self.triplet_cosine_loss(output_dict['txt_eos_prompted'], output_dict['cap_eos'])

        return ALL_LOSS

    def cal_feature_differ(self, f1, f2):
        kl = F.kl_div(F.log_softmax(f1, dim=-1), F.softmax(f2, dim=-1), reduction="none")
        per_kl = torch.mean(kl, dim=-1)
        return per_kl

    def triplet_cosine_loss(self, out_1, out_2):
        bz = out_1.size(0)
        out_1 = out_1 / torch.norm(out_1, dim=-1, keepdim=True)
        out_2 = out_2 / torch.norm(out_2, dim=-1, keepdim=True)
        scores = out_1.mm(out_2.T)
        # compute image-sentence score matrix
        diagonal = scores.diag().view(bz, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column: caption retrieval
        cost_s = (1 + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row: image retrieval
        cost_im = (1 + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        mask = mask.to(cost_s.device)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        # maximum and mean
        # cost_s_max, cost_im_max = cost_s.max(1)[0], cost_im.max(0)[0]
        cost_s_mean, cost_im_mean = cost_s.mean(1), cost_im.mean(0)

        return cost_s_mean.sum() + cost_im_mean.sum()

    def info_nce_loss(self, out_1, out_2, temperature):
        # out_*: ND
        bz = out_1.size(0)
        targets = torch.arange(bz).type_as(out_1).long()

        scores = out_1.mm(out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, targets)
        loss1 = F.cross_entropy(scores1, targets)

        return 0.5 * (loss0 + loss1)

    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        # a: ND
        # b: MD
        # label_sim: NM
        s = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)
        b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))
        return b_loss

    def quantization_loss_2(self, hash_feature, B):
        return F.mse_loss(hash_feature, B, reduction='sum') / (hash_feature.shape[0]) / self.k_bits

    def test(self):
        if self.args.pretrained == "" or self.args.pretrained == "MODEL_PATH":
            self.logger.error("test step must load a model! please set the --pretrained argument.")
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")

        self.change_state(mode="valid")

        # TODO
        test_caption_path = os.path.join(main_path, 'path/to/your/test/image_caption')
        test_image_dir = os.path.join(main_path, 'path/to/your/test/generated_image')

        q_t = self.get_text_code_LMA(test_caption_path, test_image_dir)
        q_i, _ = self.get_code(self.query_loader, self.args.query_num, split="query")
        r_i, r_t = self.get_code(self.retrieval_loader, self.args.retrieval_num, split="retrieval")

        _k_ = None
        mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), self.query_labels.to(self.device),
                            self.retrieval_labels.to(self.device), _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), self.query_labels.to(self.device),
                            self.retrieval_labels.to(self.device), _k_).item()

        self.save_mat(q_i, q_t, r_i, r_t, 0)

        self.logger.info(f"MAP(i->t): {round(mAPi2t, 5)}, MAP(t->i): {round(mAPt2i, 5)}")
        self.logger.info(">>>>>> Save *.mat data! Exit...")

