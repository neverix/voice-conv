import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.dsp import *
import time
from layers.overtone import Overtone
from layers.vector_quant import VectorQuant
from layers.downsampling_encoder import DownsamplingEncoder
import utils.env as env
import utils.logger as logger
import random


class Model(nn.Module):
    def __init__(self, rnn_dims, fc_dims, global_decoder_cond_dims, upsample_factors, normalize_vq=False,
            noise_x=False, noise_y=False):
        super().__init__()
        self.n_classes = 256
        self.overtone = Overtone(rnn_dims, fc_dims, 128, global_decoder_cond_dims)
        self.vq = VectorQuant(1, 512, 128, normalize=normalize_vq)
        self.noise_x = noise_x
        self.noise_y = noise_y
        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(128, encoder_layers)
        self.frame_advantage = 15
        self.num_params()

    def forward(self, global_decoder_cond, x, samples):
        # x: (N, 768, 3)
        #logger.log(f'x: {x.size()}')
        # samples: (N, 1022)
        #logger.log(f'samples: {samples.size()}')
        continuous = self.encoder(samples)
        # continuous: (N, 14, 64)
        #logger.log(f'continuous: {continuous.size()}')
        discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
        # discrete: (N, 14, 1, 64)
        #logger.log(f'discrete: {discrete.size()}')

        # cond: (N, 768, 64)
        #logger.log(f'cond: {cond.size()}')
        return self.overtone(x, discrete.squeeze(2), global_decoder_cond), vq_pen.mean(), encoder_pen.mean(), entropy

    def after_update(self):
        self.overtone.after_update()
        self.vq.after_update()

    def forward_generate(self, global_decoder_cond, samples, deterministic=False, use_half=False, verbose=False):
        if use_half:
            samples = samples.half()
        # samples: (L)
        #logger.log(f'samples: {samples.size()}')
        self.eval()
        with torch.no_grad() :
            continuous = self.encoder(samples)
            discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
            logger.log(f'entropy: {entropy}')
            # cond: (1, L1, 64)
            #logger.log(f'cond: {cond.size()}')
            output = self.overtone.generate(discrete.squeeze(2), global_decoder_cond, use_half=use_half, verbose=verbose)
        self.train()
        return output

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logger.log('Trainable Parameters: %.3f million' % parameters)

    def load_state_dict(self, dict, strict=True):
        if strict:
            return super().load_state_dict(self.upgrade_state_dict(dict))
        else:
            my_dict = self.state_dict()
            new_dict = {}
            for key, val in dict.items():
                if key not in my_dict:
                    logger.log(f'Ignoring {key} because no such parameter exists')
                elif val.size() != my_dict[key].size():
                    logger.log(f'Ignoring {key} because of size mismatch')
                else:
                    logger.log(f'Loading {key}')
                    new_dict[key] = val
            return super().load_state_dict(new_dict, strict=False)

    def upgrade_state_dict(self, state_dict):
        out_dict = state_dict.copy()
        return out_dict

    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if name.startswith('encoder.') or name.startswith('vq.'):
                logger.log(f'Freezing {name}')
                param.requires_grad = False
            else:
                logger.log(f'Not freezing {name}')

    def pad_left(self):
        return max(self.pad_left_decoder(), self.pad_left_encoder())

    def pad_left_decoder(self):
        return self.overtone.pad()

    def pad_left_encoder(self):
        return self.encoder.pad_left + (self.overtone.cond_pad - self.frame_advantage) * self.encoder.total_scale

    def pad_right(self):
        return self.frame_advantage * self.encoder.total_scale

    def total_scale(self):
        return self.encoder.total_scale

    def do_train(self, paths, dataset, optimiser, epochs, batch_size, step, lr=1e-4, valid_index=[], use_half=False, do_clip=False):

        if use_half:
            import apex
            optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
        for p in optimiser.param_groups : p['lr'] = lr
        criterion = nn.NLLLoss().cuda()
        k = 0
        saved_k = 0
        pad_left = self.pad_left()
        pad_left_encoder = self.pad_left_encoder()
        pad_left_decoder = self.pad_left_decoder()
        if self.noise_x:
            extra_pad_right = 127
        else:
            extra_pad_right = 0
        pad_right = self.pad_right() + extra_pad_right
        window = 16 * self.total_scale()
        logger.log(f'pad_left={pad_left_encoder}|{pad_left_decoder}, pad_right={pad_right}, total_scale={self.total_scale()}')

        for e in range(epochs) :

            trn_loader = DataLoader(dataset, collate_fn=lambda batch: env.collate_multispeaker_samples(pad_left, window, pad_right, batch), batch_size=batch_size,
                                    num_workers=2, shuffle=True, pin_memory=True)

            start = time.time()
            running_loss_c = 0.
            running_loss_f = 0.
            running_loss_vq = 0.
            running_loss_vqc = 0.
            running_entropy = 0.
            running_max_grad = 0.
            running_max_grad_name = ""

            iters = len(trn_loader)

            for i, (speaker, wave16) in enumerate(trn_loader) :

                speaker = speaker.cuda()
                wave16 = wave16.cuda()

                coarse = (wave16 + 2**15) // 256
                fine = (wave16 + 2**15) % 256

                coarse_f = coarse.float() / 127.5 - 1.
                fine_f = fine.float() / 127.5 - 1.
                total_f = (wave16.float() + 0.5) / 32767.5

                if self.noise_y:
                    noisy_f = total_f * (0.02 * torch.randn(total_f.size(0), 1).cuda()).exp() + 0.003 * torch.randn_like(total_f)
                else:
                    noisy_f = total_f

                if use_half:
                    coarse_f = coarse_f.half()
                    fine_f = fine_f.half()
                    noisy_f = noisy_f.half()

                x = torch.cat([
                    coarse_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    fine_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    coarse_f[:, pad_left-pad_left_decoder+1:1-pad_right].unsqueeze(-1),
                    ], dim=2)
                y_coarse = coarse[:, pad_left+1:1-pad_right]
                y_fine = fine[:, pad_left+1:1-pad_right]

                if self.noise_x:
                    # Randomly translate the input to the encoder to encourage
                    # translational invariance
                    total_len = coarse_f.size(1)
                    translated = []
                    for j in range(coarse_f.size(0)):
                        shift = random.randrange(256) - 128
                        translated.append(noisy_f[j, pad_left-pad_left_encoder+shift:total_len-extra_pad_right+shift])
                    translated = torch.stack(translated, dim=0)
                else:
                    translated = noisy_f[:, pad_left-pad_left_encoder:]
                p_cf, vq_pen, encoder_pen, entropy = self(speaker, x, translated)
                p_c, p_f = p_cf
                loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
                loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
                encoder_weight = 0.01 * min(1, max(0.1, step / 1000 - 1))
                loss = loss_c + loss_f + vq_pen + encoder_weight * encoder_pen

                optimiser.zero_grad()
                if use_half:
                    optimiser.backward(loss)
                    if do_clip:
                        raise RuntimeError("clipping in half precision is not implemented yet")
                else:
                    loss.backward()
                    if do_clip:
                        max_grad = 0
                        max_grad_name = ""
                        for name, param in self.named_parameters():
                            if param.grad is not None:
                                param_max_grad = param.grad.data.abs().max()
                                if param_max_grad > max_grad:
                                    max_grad = param_max_grad
                                    max_grad_name = name
                                if 1000000 < param_max_grad:
                                    logger.log(f'Very large gradient at {name}: {param_max_grad}')
                        if 100 < max_grad:
                            for param in self.parameters():
                                if param.grad is not None:
                                    if 1000000 < max_grad:
                                        param.grad.data.zero_()
                                    else:
                                        param.grad.data.mul_(100 / max_grad)
                        if running_max_grad < max_grad:
                            running_max_grad = max_grad
                            running_max_grad_name = max_grad_name

                        if 100000 < max_grad:
                            torch.save(self.state_dict(), "bad_model.pyt")
                            raise RuntimeError("Aborting due to crazy gradient (model saved to bad_model.pyt)")
                optimiser.step()
                running_loss_c += loss_c.item()
                running_loss_f += loss_f.item()
                running_loss_vq += vq_pen.item()
                running_loss_vqc += encoder_pen.item()
                running_entropy += entropy

                self.after_update()

                speed = (i + 1) / (time.time() - start)
                avg_loss_c = running_loss_c / (i + 1)
                avg_loss_f = running_loss_f / (i + 1)
                avg_loss_vq = running_loss_vq / (i + 1)
                avg_loss_vqc = running_loss_vqc / (i + 1)
                avg_entropy = running_entropy / (i + 1)

                step += 1
                k = step // 1000
                logger.status(f'Epoch: {e+1}/{epochs} -- Batch: {i+1}/{iters} -- Loss: c={avg_loss_c:#.4} f={avg_loss_f:#.4} vq={avg_loss_vq:#.4} vqc={avg_loss_vqc:#.4} -- Entropy: {avg_entropy:#.4} -- Grad: {running_max_grad:#.1} {running_max_grad_name} Speed: {speed:#.4} steps/sec -- Step: {k}k ')

            os.makedirs(paths.checkpoint_dir, exist_ok=True)
            torch.save(self.state_dict(), paths.model_path())
            np.save(paths.step_path(), step)
            logger.log_current_status()
            logger.log(f' <saved>; w[0][0] = {self.overtone.wavernn.gru.weight_ih_l0[0][0]}')
            if k > saved_k + 15:
                torch.save(self.state_dict(), paths.model_hist_path(step))
                saved_k = k
                self.do_generate(paths, step, dataset.path, valid_index)

    def do_generate(self, paths, step, data_path, test_index, deterministic=False, use_half=False, verbose=False):
        k = step // 1000
        dataset = env.MultispeakerDataset(test_index, data_path)
        loader = DataLoader(dataset, shuffle=False)
        data = [x for x in loader]
        n_points = len(data)
        gt = [(x[0].float() + 0.5) / (2**15 - 0.5) for speaker, x in data]
        extended = [np.concatenate([np.zeros(self.pad_left_encoder(), dtype=np.float32), x, np.zeros(self.pad_right(), dtype=np.float32)]) for x in gt]
        speakers = [torch.FloatTensor(speaker[0].float()) for speaker, x in data]
        maxlen = max([len(x) for x in extended])
        aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen-len(x))]) for x in extended]
        os.makedirs(paths.gen_path(), exist_ok=True)
        out = self.forward_generate(torch.stack(speakers + list(reversed(speakers)), dim=0).cuda(), torch.stack(aligned + aligned, dim=0).cuda(), verbose=verbose, use_half=use_half)
        logger.log(f'out: {out.size()}')
        for i, x in enumerate(gt) :
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x.cpu().numpy(), sr=sample_rate)
            audio = out[i][:len(x)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
            audio_tr = out[n_points+i][:len(x)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_transferred.wav', audio_tr, sr=sample_rate)
