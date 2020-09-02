import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, pdb, copy, os, math, numpy, copy, time
import utils
from utils import assert_shape


####################
# Basic modules
####################

# encodes a sequence of input frames and states, and optionally a cost or action, to a hidden representation
class encoder(nn.Module):
    def __init__(self, opt, a_size, n_inputs, states=True, state_input_size=4, n_channels=3):
        super(encoder, self).__init__()
        self.opt = opt
        self.a_size = a_size
        self.n_inputs = opt.ncond if n_inputs is None else n_inputs
        self.n_channels = n_channels
        # frame encoder
        if opt.layers == 3:
            assert(opt.nfeature % 4 == 0)
            self.feature_maps = (opt.nfeature // 4, opt.nfeature // 2, opt.nfeature)
            self.f_encoder = nn.Sequential(
                nn.Conv2d(n_channels * self.n_inputs, self.feature_maps[0], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
            )
        elif opt.layers == 4:
            assert(opt.nfeature % 8 == 0)
            self.feature_maps = (opt.nfeature // 8, opt.nfeature // 4, opt.nfeature // 2, opt.nfeature)
            self.f_encoder = nn.Sequential(
                nn.Conv2d(n_channels * self.n_inputs, self.feature_maps[0], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[2], self.feature_maps[3], 4, 2, 1)
            )

        if states:
            n_hidden = self.feature_maps[-1]
            # state encoder
            self.s_encoder = nn.Sequential(
                nn.Linear(state_input_size * self.n_inputs, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, opt.hidden_size)
            )

        if a_size > 0:
            # action or cost encoder
            n_hidden = self.feature_maps[-1]
            self.a_encoder = nn.Sequential(
                nn.Linear(a_size, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, opt.hidden_size)
            )

    def forward(self, images, states=None, actions=None):
        bsize = images.size(0)
        h = self.f_encoder(images.view(bsize, self.n_inputs * self.n_channels, self.opt.height, self.opt.width))
        if states is not None:
            h = h + self.s_encoder(states.contiguous().view(bsize, -1)).view(h.size())
        if actions is not None:
            a = self.a_encoder(actions.contiguous().view(bsize, self.a_size))
            h = h + a.view(h.size())
        return h


class u_network(nn.Module):
    def __init__(self, opt):
        super(u_network, self).__init__()
        self.opt = opt
        self.encoder = nn.Sequential(
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 4, 2, 1),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Zeming says: wtf this means the net doesn't even see every pixel? kernel 1 stride 2???
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, (4, 1), 2, 1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.opt.nfeature, self.opt.nfeature, (4, 1), 2, 1),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.opt.nfeature, self.opt.nfeature, (4, 3), 2, 0)
        )

        assert(self.opt.layers == 3) # hardcoded sizes
        self.hidden_size = self.opt.nfeature*3*2
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.hidden_size)
        )

    def forward(self, h):
        h1 = self.encoder(h)
        h2 = self.fc(h1.view(-1, self.hidden_size))
        h2 = h2.view(h1.size())
        h3 = self.decoder(h2)
        return h3


# decodes a hidden state into a predicted frame, a predicted state and a predicted cost vector
class decoder(nn.Module):
    def __init__(self, opt, n_out=1):
        super(decoder, self).__init__()
        self.opt = opt
        self.n_out = n_out
        if self.opt.layers == 3:
            assert(opt.nfeature % 4 == 0)
            self.feature_maps = [int(opt.nfeature/4), int(opt.nfeature/2), opt.nfeature]
            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.feature_maps[2], self.feature_maps[1], (4, 4), 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[1], self.feature_maps[0], (5, 5), 2, (0, 1)),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[0], self.n_out*3, (2, 2), 2, (0, 1))
            )

            self.h_reducer = nn.Sequential(
                nn.Conv2d(self.feature_maps[2], self.feature_maps[2], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[2], self.feature_maps[2], (4, 1), (2, 1), 0),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

        elif self.opt.layers == 4:
            assert(opt.nfeature % 8 == 0)
            self.feature_maps = [int(opt.nfeature/8), int(opt.nfeature/4), int(opt.nfeature/2), opt.nfeature]

            self.f_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.feature_maps[3], self.feature_maps[2], (4, 4), 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[2], self.feature_maps[1], (5, 5), 2, (0, 1)),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[1], self.feature_maps[0], (2, 4), 2, (1, 0)),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.feature_maps[0], self.n_out*3, (2, 2), 2, (1, 0))
            )

            self.h_reducer = nn.Sequential(
                nn.Conv2d(opt.nfeature, opt.nfeature, (4, 1), (2, 1), 0),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

        n_hidden = self.feature_maps[-1]

        self.s_predictor = nn.Sequential(
            nn.Linear(2*n_hidden, n_hidden),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, self.n_out*4)
        )

    def forward(self, h):
        bsize = h.size(0)
        h = h.view(bsize, self.feature_maps[-1], self.opt.h_height, self.opt.h_width)
        h_reduced = self.h_reducer(h).view(bsize, -1)
        pred_state = self.s_predictor(h_reduced)
        pred_image = self.f_decoder(h)
        pred_image = pred_image[:, :, :self.opt.height, :self.opt.width].clone()
        pred_image = pred_image.view(bsize, 1, 3*self.n_out, self.opt.height, self.opt.width)
        return pred_image, pred_state


# expands a latent variable to the size of the hidden representation
class z_expander(nn.Module):
    def __init__(self, opt, n_steps):
        super(z_expander, self).__init__()
        self.opt = opt
        self.n_steps = n_steps
        self.z_expander = nn.Sequential(
            nn.Linear(opt.nz, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, n_steps * opt.nfeature * self.opt.h_height * self.opt.h_width)
        )

    def forward(self, z):
        bsize = z.size(0)
        z_exp = self.z_expander(z).view(bsize, self.n_steps, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
        return z_exp


# maps a hidden representation to a distribution over latent variables.
# We use this for the VAE models.
class z_network_gaussian(nn.Module):
    def __init__(self, opt):
        super(z_network_gaussian, self).__init__()
        self.opt = opt

        self.network = nn.Sequential(
            nn.Linear(opt.nfeature*self.opt.h_height*self.opt.h_width, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, 2*opt.nz)
        )

    def reparameterize(self, mu, logvar, sample):
        if self.training or sample:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, inputs):
        bsize = inputs.size(0)
        inputs = inputs.view(bsize, self.opt.nfeature*self.opt.h_height*self.opt.h_width)
        z_params = self.network(inputs).view(-1, self.opt.nz, 2)
        mu = z_params[:, :, 0]
        logvar = z_params[:, :, 1]
        return mu, logvar

    def forward(self, inputs, sample=True):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar, sample)
        return z, mu, logvar


# takes as input a sequence of frames, states and actions, and outputs the parameters of a
# Gaussian Mixture Model.
class PriorMDN(nn.Module):
    def __init__(self, opt):
        super(PriorMDN, self).__init__()
        self.opt = opt
        self.n_inputs = opt.ncond
        self.encoder = encoder(opt, 0, opt.ncond)

        self.network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.pi_net = nn.Linear(opt.n_hidden, opt.n_mixture)
        self.mu_net = nn.Linear(opt.n_hidden, opt.n_mixture*opt.nz)
        self.sigma_net = nn.Linear(opt.n_hidden, opt.n_mixture*opt.nz)

    def forward(self, input_images, input_states):
        bsize = input_images.size(0)
        h = self.encoder(input_images, input_states).view(bsize, -1)
        h = self.network(h)
        pi = F.softmax(self.pi_net(h), dim=1)
        mu = self.mu_net(h).view(bsize, self.opt.n_mixture, self.opt.nz)
        sigma = F.softplus(self.sigma_net(h)).view(bsize, self.opt.n_mixture, self.opt.nz)
        sigma = torch.clamp(sigma, min=1e-3)
        return pi, mu, sigma

    # First extract z vectors by passing inputs, actions and targets through an external model, and uses these as
    # targets. Useful for training the prior network to predict the z vectors inferred by a previously trained
    # forward model.
    def forward_thru_model(self, model, inputs, actions, targets):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        npred = actions.size(1)
        ploss = torch.zeros(1).cuda()

        for t in range(npred):
            h_x = model.encoder(input_images, input_states)
            target_images, target_states, target_costs = targets
            h_y = model.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
            z = model.z_network((h_x + h_y).view(bsize, -1))
            pi, mu, sigma = self(input_images, input_states)
            # prior loss
            ploss += utils.mdn_loss_fn(pi, sigma, mu, z)
            z_exp = model.z_expander(z).view(bsize, model.opt.nfeature, model.opt.h_height, model.opt.h_width)
            h_x = h_x.view(bsize, model.opt.nfeature, model.opt.h_height, model.opt.h_width)
            a_emb = model.a_encoder(actions[:, t]).view(h_x.size())
            h = h_x + z_exp
            h = h + a_emb
            h = h + model.u_network(h)
            pred_image, pred_state, pred_cost = model.decoder(h)

            pred_image.detach()
            pred_state.detach()
            pred_cost.detach()
            pred_image = torch.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
            pred_state = pred_state + input_states[:, -1]
            # pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)

        return ploss / npred


# takes as input a sequence of frames, states and actions, and outputs the parameters of a
# Gaussian Mixture Model.
class PriorGaussian(nn.Module):
    def __init__(self, opt, nz):
        super(PriorGaussian, self).__init__()
        self.opt = opt
        self.n_inputs = opt.ncond
        self.encoder = encoder(opt, 0, opt.ncond)
        self.nz = nz

        self.network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mu_net = nn.Linear(opt.n_hidden, nz)
        self.sigma_net = nn.Linear(opt.n_hidden, nz)

    def forward(self, input_images, input_states, normalize_inputs=False, normalize_outputs=False, n_samples=1):
        if normalize_inputs:
            input_images = input_images.clone().float().div_(255.0)
            input_states -= self.stats['s_mean'].view(1, 4).expand(input_states.size())
            input_states /= self.stats['s_std'].view(1, 4).expand(input_states.size())
            input_images = input_images.cuda().unsqueeze(0)
            input_states = input_states.cuda().unsqueeze(0)

        bsize = input_images.size(0)
        h = self.encoder(input_images, input_states).view(bsize, -1)
        h = self.network(h)
        mu = self.mu_net(h).view(bsize, self.nz)
        sigma = F.softplus(self.sigma_net(h)).view(bsize, self.nz)
        sigma = torch.clamp(sigma, min=1e-3)

        eps = torch.randn(bsize, n_samples, self.opt.n_actions).cuda()
        a = eps * sigma.view(bsize, 1, self.opt.n_actions)
        a = a + mu.view(bsize, 1, self.opt.n_actions)

        if normalize_outputs:
            a = a.data
            a *= self.stats['a_std'].view(1, 1, 2).expand(a.size()).cuda()
            a += self.stats['a_mean'].view(1, 1, 2).expand(a.size()).cuda()

        return mu, sigma, a


# Mixture Density network (fully-connected).
class v_network_mdn_fc(nn.Module):
    def __init__(self, opt, n_outputs):
        super(v_network_mdn_fc, self).__init__()
        self.opt = opt
        self.n_outputs = n_outputs

        self.network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.pi_net = nn.Linear(opt.nfeature, opt.n_mixture)
        self.mu_net = nn.Linear(opt.nfeature, opt.n_mixture*n_outputs)
        self.sigma_net = nn.Linear(opt.nfeature, opt.n_mixture*n_outputs)

    def forward(self, h):
        bsize = h.size(0)
        h = h.view(bsize, self.opt.hidden_size)
        h = self.network(h)
        pi = F.softmax(self.pi_net(h), dim=1)
        mu = self.mu_net(h).view(bsize, self.opt.n_mixture, self.n_outputs)
        sigma = F.softplus(self.sigma_net(h)).view(bsize, self.opt.n_mixture, self.n_outputs)
        sigma = torch.clamp(sigma, min=1e-3)
        return pi, mu, sigma


class v_network(nn.Module):
    def __init__(self, opt):
        super(v_network, self).__init__()
        self.opt = opt

        self.network = nn.Sequential(
            nn.Linear(self.opt.hidden_size, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nz)
        )

    def forward(self, h):
        bsize = h.size(0)
        h = h.view(bsize, self.opt.hidden_size)
        u = self.network(h)
        u = u / torch.norm(u, 2, 1).unsqueeze(1)
        return u


# combines a sequence of images with the state vector.
class policy_encoder(nn.Module):
    def __init__(self, opt, n_channels=4):
        super(policy_encoder, self).__init__()
        self.opt = opt
        self.n_channels = n_channels

        self.convnet = nn.Sequential(
            nn.Conv2d(n_channels * opt.ncond, opt.nfeature, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(opt.nfeature, opt.nfeature, 4, 2, 1),
            nn.ReLU()
        )

        self.embed = nn.Sequential(
            nn.Linear(opt.ncond*opt.n_inputs, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden)
        )

        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width

    def forward(self, state_images, states):
        bsize = state_images.size(0)
        state_images = state_images.view(bsize, self.n_channels * self.opt.ncond, self.opt.height, self.opt.width)
        states = states.view(bsize, -1)
        hi = self.convnet(state_images).view(bsize, self.hsize)
        hs = self.embed(states)
        h = torch.cat((hi, hs), 1)
        return h


###############
# Main models
###############

# forward model, deterministic (compatible with TEN3 model, use to initialize)
class FwdCNN(nn.Module):
    def __init__(self, opt, mfile):
        super(FwdCNN, self).__init__()
        self.opt = opt
        # If we are given a model file, use it to initialize this model.
        # otherwise initialize from scratch
        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)
            self.a_encoder = nn.Sequential(
                nn.Linear(self.opt.n_actions, self.opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.opt.nfeature, self.opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.opt.nfeature, self.opt.hidden_size)
            )
            self.u_network = u_network(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)['model']
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond

    # dummy function
    def sample_z(self, bsize, method=None):
        return torch.zeros(bsize, 32).cuda()

    def forward_single_step(self, input_images, input_states, action, z):
        # encode the inputs (without the action)
        bsize = input_images.size(0)
        h_x = self.encoder(input_images, input_states)
        h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
        a_emb = self.a_encoder(action).view(h_x.size())

        h = h_x
        h = h + a_emb
        h = h + self.u_network(h)
        pred_image, pred_state = self.decoder(h)
        pred_image = torch.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
        pred_state = pred_state + input_states[:, -1]
        return pred_image, pred_state

    def forward(self, inputs, actions, target, sampling=None, z_dropout=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states = [], []
        for t in range(npred):
            h = self.encoder(input_images, input_states)
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = h + a_emb
            h = h + self.u_network(h)
            pred_image, pred_state = self.decoder(h)
            pred_image = torch.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
#            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            pred_state = pred_state + input_states[:, -1]
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        return [pred_images, pred_states, None], torch.zeros(1).cuda()

    def create_policy_net(self, opt):
        if opt.policy == 'policy-gauss':
            self.policy_net = StochasticPolicy(opt)
        if opt.policy == 'policy-ten':
            self.policy_net = PolicyTEN(opt)
        elif opt.policy == 'policy-vae':
            self.policy_net = PolicyVAE(opt)


# this version adds the actions *after* the z variables
class FwdCNN_VAE(nn.Module):
    def __init__(self, opt, mfile=''):
        super(FwdCNN_VAE, self).__init__()
        self.opt = opt

        if mfile == '':
            self.encoder = encoder(opt, 0, opt.ncond)
            self.decoder = decoder(opt)
            self.a_encoder = nn.Sequential(
                nn.Linear(self.opt.n_actions, self.opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.opt.nfeature, self.opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.opt.nfeature, self.opt.hidden_size)
            )
            self.u_network = u_network(opt)
        else:
            print('[initializing encoder and decoder with: {}]'.format(mfile))
            self.mfile = mfile
            pretrained_model = torch.load(mfile)
            if type(pretrained_model) is dict: pretrained_model = pretrained_model['model']
            self.encoder = pretrained_model.encoder
            self.decoder = pretrained_model.decoder
            self.a_encoder = pretrained_model.a_encoder
            self.u_network = pretrained_model.u_network
            self.encoder.n_inputs = opt.ncond
            self.decoder.n_out = 1

        self.y_encoder = encoder(opt, 0, 1, states=False)

        self.z_network = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, 2*opt.nz)
        )

        if self.opt.model == 'fwd-cnn-vae3-lp':
            self.z_network_prior = nn.Sequential(
                nn.Linear(opt.hidden_size, opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, opt.nfeature),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.nfeature, 2*opt.nz)
            )

        self.z_zero = torch.zeros(self.opt.batch_size, self.opt.nz)
        self.z_expander = nn.Linear(opt.nz, opt.hidden_size)

    def reparameterize(self, mu, logvar, sample):
        if self.training or sample:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample_z(self, bsize, method=None, h_x=None):
        if self.opt.model == 'fwd-cnn-vae-fp':
            z = torch.randn(bsize, self.opt.nz).cuda()
        elif self.opt.model == 'fwd-cnn-vae-lp':
            mu_logvar_prior = self.z_network_prior(h_x.view(bsize, -1)).view(bsize, 2, self.opt.nz)
            mu_prior = mu_logvar_prior[:, 0]
            logvar_prior = mu_logvar_prior[:, 1]
            z = self.reparameterize(mu_prior, logvar_prior, True)
        return z

    def forward_single_step(self, input_images, input_states, action, z):
        # encode the inputs (without the action)
        bsize = input_images.size(0)
        h_x = self.encoder(input_images, input_states)
        z_exp = self.z_expander(z).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
        h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
        a_emb = self.a_encoder(action).view(h_x.size())

        h = h_x + z_exp
        h = h + a_emb
        h = h + self.u_network(h)
        pred_image, pred_state = self.decoder(h)
        pred_image = torch.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
        # pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
        pred_state = pred_state + input_states[:, -1]

        return pred_image, pred_state

    def forward(self, inputs, actions, targets, save_z=False, sampling=None, z_dropout=0.0, z_seq=None, noise=None):
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.opt.n_actions)
        npred = actions.size(1)
        ploss = torch.zeros(1).cuda()
        ploss2 = torch.zeros(1).cuda()

        pred_images, pred_states = [], []
        z_list = []

        z = None
        for t in range(npred):
            # encode the inputs (without the action)
            h_x = self.encoder(input_images, input_states)
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states, _ = targets
                # encode the targets into z
                h_y = self.y_encoder(target_images[:, t].unsqueeze(1).contiguous())
                if random.random() < z_dropout:
                    z = self.sample_z(bsize, method=None, h_x=h_x).data
                else:
                    mu_logvar = self.z_network((h_x + h_y).view(bsize, -1)).view(bsize, 2, self.opt.nz)
                    mu = mu_logvar[:, 0]
                    logvar = mu_logvar[:, 1]
                    z = self.reparameterize(mu, logvar, True)
                    logvar = torch.clamp(logvar, max=4)  # this can go to inf when taking exp(), so clamp it
                    if self.opt.model == 'fwd-cnn-vae-fp':
                        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        kld /= bsize
                        ploss += kld
                    else:
                        raise ValueError
            else:
                if z_seq is not None:
                    z = z_seq[t]
                else:
                    z = self.sample_z(bsize, method=None, h_x=h_x)

            z_list.append(z)
            z_exp = self.z_expander(z).view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
            h = h_x + z_exp
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = h + a_emb
            h = h + self.u_network(h)

            pred_image, pred_state = self.decoder(h)
            if sampling is not None:
                pred_image.detach()
                pred_state.detach()
            pred_image = torch.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
            pred_state = pred_state + input_states[:, -1]

            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        z_list = torch.stack(z_list, 1)
        return [pred_images, pred_states, z_list], [ploss, ploss2]

    def reset_action_buffer(self, npred):
        self.actions_buffer = torch.zeros(npred, self.opt.n_actions).cuda()
        self.optimizer_a_stats = None

    def create_policy_net(self, opt):
        if opt.policy == 'policy-gauss':
            self.policy_net = StochasticPolicy(opt)
        if opt.policy == 'policy-ten':
            self.policy_net = PolicyTEN(opt)
        elif opt.policy == 'policy-vae':
            self.policy_net = PolicyVAE(opt)
        elif opt.policy == 'policy-deterministic':
            self.policy_net = DeterministicPolicy(opt)

    def create_prior_net(self, opt):
        self.prior_net = PriorGaussian(opt, opt.context_dim)

    def intype(self, t):
        if t == 'gpu':
            self.cuda()
            self.z_zero = self.z_zero.cuda()
            self.use_cuda = True
        elif t == 'cpu':
            self.cpu()
            self.use_cuda = False
            self.z_zero = self.z_zero.cpu()


class Multiagent_u_net(nn.Module):
    '''
    This the u network should handle the entire receptive field, we just make it
    one more layer...
    '''
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.encoder = nn.Sequential(
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, (7, 2), 2),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, (6, 2), 2),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, (4, 2), 2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.opt.nfeature, self.opt.nfeature, (4, 2), 2),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.opt.nfeature, self.opt.nfeature, (6, 2), 2),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.opt.nfeature, self.opt.nfeature, (7, 2), 2),
            nn.Dropout2d(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, 3, 1, 1)
        )

        assert(self.opt.layers == 3) # hardcoded sizes
        assert(self.opt.height == 117)
        assert(self.opt.width == 24)
        self.hidden_size = self.opt.nfeature*12*3
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.hidden_size)
        )

    def forward(self, h):
        h1 = self.encoder(h)
        h2 = self.fc(h1.view(-1, self.hidden_size))
        h2 = h2.view(h1.size())
        h3 = self.decoder(h2)
        assert h3.shape == h.shape
        return h3


class MultiagentFwdObj(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.T = opt.ncond
        self.B = opt.batch_size
        self.nStates = 9
        ichan = self.T * 3
        schan = self.T * self.nStates
        self.feature_maps = (opt.nfeature // 4, opt.nfeature // 2, opt.nfeature)
        self.image_conv = nn.Sequential(
                nn.Conv2d(ichan, opt.nfeature, 3, 1, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(opt.nfeature, opt.nfeature, 3, 1, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(opt.nfeature, opt.nfeature, 3, 1, 1),
            )

        self.agent_emb = nn.Sequential(
            nn.Linear(schan, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature)
        )
        self.matf_conv_trunk = nn.Sequential(
            # The concatented features are probably very differing in norms
            # So we first do a group norm before sending it to the u_network
            nn.GroupNorm(2, opt.nfeature*2),
            nn.Conv2d(2 * opt.nfeature, opt.nfeature, 1, 1), # MLP
            nn.LeakyReLU(0.2, inplace=True),
            Multiagent_u_net(opt)
        )
        self.matf_conv_img = nn.Sequential(
            nn.Conv2d(opt.nfeature, opt.nfeature, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.nfeature, 3, 3, 1, 1),
        )
        self.matf_conv_agent = nn.Sequential(
            nn.Conv2d(opt.nfeature, opt.nfeature, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.nfeature, opt.nfeature, 3, 1, 1),
        )
        self.agent_mlp = nn.Sequential(
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nfeature, opt.nfeature)
        )
        self.trajectory_prediction = nn.Sequential(
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.opt.nfeature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.opt.nfeature, self.nStates)
        )
        self.z_linear = nn.Linear(opt.nz, 2 * opt.nfeature)

    def forward(self, input_images, input_states, action, z):
        B, T, nRGB, H, W = assert_shape(input_images, -1, -1, -1, -1, -1)
        nCars, nStates = assert_shape(input_states, B, T, -1, -1)
        nActions = assert_shape(action, B, T, -1)
        assert_shape(z, B, self.opt.nz)

        # Stack all timesteps together
        images = input_images.reshape(B, -1, H, W)
        # (B x nCars) x (T x nStates)
        states = input_states.transpose(1, 2).reshape(-1, T * nStates)

        # In the most recent timestep, which cars are in frame
        last_frame_mask = input_states[:, -1, :, 8].unsqueeze(-1)
        xy_locs = input_states[:, -1, :, 0:2].round()
        # Get rid of all the invalid locations (fill them with -1)
        xy_locs = xy_locs * last_frame_mask - (1 - last_frame_mask) 

        image_emb = self.image_conv(images) # B x C x H x W
        agent_emb = self.agent_emb(states).reshape(B, nCars, -1) # B x nCars x C
        scattered = torch.cat([utils.scatter_sum_2d(xy_locs, agent_emb, input_images.shape[-2:]), image_emb], dim=1)

        new_z = self.z_linear(z).reshape(B, 2 * self.opt.nfeature, 1, 1)
        scattered += new_z

        # Here, we scattered all the elements to the correct 2d locations, so we do a u-net conv trunk
        trunk = self.matf_conv_trunk(scattered)
        spatial = self.matf_conv_img(trunk)  # B x 3 x H x W
        agent_res = self.agent_mlp(utils.gather_2d(xy_locs, self.matf_conv_agent(trunk)))

        assert_shape(agent_res, *agent_emb.shape)
        assert_shape(spatial, B, 3, H, W)
        assert_shape(trunk, B, -1, H, W)

        agent_emb += agent_res
        agent_out = self.trajectory_prediction(agent_emb)

        # XXX don't predict length, width, mask components, copy them in
        # Some assertions to remind ourselves to catch mistakes if we make them:
        assert_shape(agent_out, B, nCars, 9)
        assert input_states[:, :, 0, 0].allclose(input_states.new_tensor(W/2, dtype=input_states.dtype)), "Location of ego car should be center of frame"
        # TODO Why isn't this the case?
        # assert input_states[:, :, 0, 1].allclose(input_states.new_tensor(H/2, dtype=input_states.dtype)), "Location of ego car should be center of frame"
        assert input_states[:, :, 0, 8].allclose(input_states.new_tensor(1, dtype=input_states.dtype)), "Ego car should always be counted in mask"

        # Actually, since we are predicting a difference, just never "move" the x, y, and mask
        agent_out[:, :, 0].zero_()
        agent_out[:, :, 1].zero_()
        agent_out[:, :, 8].zero_()
        # TODO: Draw the right boxes using the agent representation...
        return spatial, agent_out


class FwdObj(nn.Module):
    def __init__(self, opt, mfile):
        super().__init__()
        self.opt = opt
        assert mfile == '', "We don't support warm starts"
        self.model = MultiagentFwdObj(opt)
        torch.set_anomaly_enabled(True) # FIXME: delete when it works

    # dummy function
    def sample_z(self, bsize, method=None):
        return torch.zeros(bsize, 32).cuda()

    def forward_single_step(self, input_images, input_states, action, z):
        # encode the inputs (without the action)
        raise NotImplementedError("Fix this function")
        bsize = input_images.size(0)
        h_x = self.encoder(input_images, input_states)
        h_x = h_x.view(bsize, self.opt.nfeature, self.opt.h_height, self.opt.h_width)
        a_emb = self.a_encoder(action).view(h_x.size())

        h = h_x
        h = h + a_emb
        h = h + self.u_network(h)
        pred_image, pred_state = self.decoder(h)
        pred_image = torch.sigmoid(pred_image + input_images[:, -1].unsqueeze(1))
        pred_state = pred_state + input_states[:, -1]
        return pred_image, pred_state

    def forward(self, inputs, actions, target, sampling=None, z_dropout=None):
        input_images, input_states = inputs
        pred_images, pred_states = [], []
        target_images, target_states, _ = target

        B, Tseen, nRGB, H, W = assert_shape(input_images, -1, -1, -1, -1, -1)
        nCars, nStates = assert_shape(input_states, B, Tseen, -1, -1)
        Tpred, nActions = assert_shape(actions, B, -1, -1)
        assert_shape(target_images, B, Tpred, nRGB, H, W)
        assert_shape(target_states, B, Tpred, nCars, nStates)

        # We run the model some number of times, hoping that dropout's enabled
        # so we get different answers
        for t in range(Tpred):
            z = input_images.new_zeros(B, self.opt.nz)  # TODO FIX
            pred_image, pred_state = self.model(input_images, input_states, actions, z)

            pred_image = torch.sigmoid(pred_image + input_images[:, -1]).unsqueeze(1)
            # since these are normalized, we are clamping to 6 standard deviations (if gaussian)
#            pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
            pred_state = pred_state + input_states[:, -1]

            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            pred_images.append(pred_image)
            pred_states.append(pred_state)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        return [pred_images, pred_states, None], torch.zeros(1).cuda()

    def create_policy_net(self, opt):
        raise NotImplementedError("Fix this function")
        if opt.policy == 'policy-gauss':
            self.policy_net = StochasticPolicy(opt)
        if opt.policy == 'policy-ten':
            self.policy_net = PolicyTEN(opt)
        elif opt.policy == 'policy-vae':
            self.policy_net = PolicyVAE(opt)



#######################################
# Policy Networks
#######################################

# deterministic CNN model
class PolicyCNN(nn.Module):
    def __init__(self, opt):
        super(PolicyCNN, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt)
        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width
        self.fc = nn.Sequential(
            nn.Linear(self.hsize + opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.npred*opt.n_actions)
        )

    def forward(self, state_images, states, actions):
        bsize = state_images.size(0)
        h = self.encoder(state_images, states)
        a = self.fc(h)
        a = a.view(bsize, self.opt.npred, self.opt.n_actions)
        return a, torch.zeros(1)


class CostPredictor(nn.Module):
    def __init__(self, opt):
        super(CostPredictor, self).__init__()
        self.opt = opt
        self.encoder = encoder(opt, 0, 1)
        self.hsize = opt.nfeature*self.opt.h_height*self.opt.h_width
        self.proj = nn.Linear(self.hsize, opt.n_hidden)

        self.fc = nn.Sequential(
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, 2),
            nn.Tanh()
        )

    def forward(self, state_images, states):
        bsize = state_images.size(0)
        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.proj(h)
        h = self.fc(h)
        return h


# Stochastic Policy, output is a diagonal Gaussian and learning uses the re-parametrization trick.
class StochasticPolicy(nn.Module):
    def __init__(self, opt, context_dim=0, actor_critic=False, output_dim=None, n_channels=4):
        super().__init__()
        self.opt = opt
        self.n_channels = n_channels
        self.encoder = encoder(opt, a_size=0, n_inputs=opt.ncond)
        self.n_outputs = opt.n_actions if output_dim is None else output_dim
        self.hsize = opt.nfeature * self.opt.h_height * self.opt.h_width
        self.proj = nn.Linear(self.hsize, opt.n_hidden)
        self.context_dim = context_dim

        self.fc = nn.Sequential(
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden)
        )

        if context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden)
            )

        self.mu_net = nn.Linear(opt.n_hidden, self.n_outputs)
        self.logvar_net = nn.Linear(opt.n_hidden, self.n_outputs)
        self.actor_critic = actor_critic
        if actor_critic:
            self.value_net = nn.Linear(opt.n_hidden, 1)
            self.saved_actions = []
            self.rewards = []

    def forward(self, state_images, states, context=None, sample=True,
                normalize_inputs=False, normalize_outputs=False, n_samples=1, std_mult=1.0):

        if normalize_inputs:
            state_images = state_images.clone().float().div_(255.0)
            states -= self.stats['s_mean'].view(1, 4).expand(states.size())
            states /= self.stats['s_std'].view(1, 4).expand(states.size())
            if state_images.dim() == 4:  # if processing single vehicle
                state_images = state_images.cuda().unsqueeze(0)
                states = states.cuda().unsqueeze(0)

        bsize = state_images.size(0)

        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.proj(h)
        if self.context_dim > 0:
            assert(context is not None)
            h = h + self.context_encoder(context)
        h = self.fc(h)
        mu = self.mu_net(h).view(bsize, self.n_outputs)
        logvar = self.logvar_net(h).view(bsize, self.n_outputs)
        logvar = torch.clamp(logvar, max=4.0)
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(bsize, n_samples, self.n_outputs).cuda()  # .cuda() is FUCKING wrong!
        a = eps * std.view(bsize, 1, self.n_outputs) * std_mult
        a = a + mu.view(bsize, 1, self.n_outputs)
        # a = 3 * torch.tanh(a)

        if normalize_outputs:  # done only at inference time, if only "volatile" was still a thing...
            a = a.data
            a.clamp_(-3, 3)
            a *= self.stats['a_std'].view(1, 1, 2).expand(a.size()).cuda()
            a += self.stats['a_mean'].view(1, 1, 2).expand(a.size()).cuda()

        entropy = std.mean()  # TODO: Fix bug! Missing ".log_()"!
        if self.actor_critic:
            value = self.value_net(h).view(bsize, 1)
            return a.squeeze(), entropy, mu, std, value
        else:
            return a.squeeze(), entropy, mu, std


class DeterministicPolicy(nn.Module):
    def __init__(self, opt, context_dim=0, output_dim=None, n_channels=4):
        super().__init__()
        self.opt = opt
        self.n_channels = n_channels
        self.encoder = encoder(opt, a_size=0, n_inputs=opt.ncond, n_channels=n_channels)
        self.n_outputs = opt.n_actions if output_dim is None else output_dim
        self.hsize = opt.nfeature * self.opt.h_height * self.opt.h_width
        self.proj = nn.Linear(self.hsize, opt.n_hidden)
        self.context_dim = context_dim

        self.fc = nn.Sequential(
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, self.n_outputs)
        )

        if context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden),
                nn.ReLU(),
                nn.Linear(opt.n_hidden, opt.n_hidden)
            )

    def forward(self, state_images, states, context=None, sample=True,
                normalize_inputs=False, normalize_outputs=False, n_samples=1):

        if normalize_inputs:
            state_images = state_images.clone().float().div_(255.0)
            states -= self.stats['s_mean'].view(1, 4).expand(states.size())
            states /= self.stats['s_std'].view(1, 4).expand(states.size())
            if state_images.dim() == 4:  # if processing single vehicle
                state_images = state_images.cuda().unsqueeze(0)
                states = states.cuda().unsqueeze(0)

        bsize = state_images.size(0)

        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.proj(h)  # from hidden_size to n_hidden
        if self.context_dim > 0:
            assert(context is not None)
            h = h + self.context_encoder(context)
        a = self.fc(h).view(bsize, self.n_outputs)

        if normalize_outputs:  # done only at inference time, if only "volatile" was still a thing...
            a = a.data
            a.clamp_(-3, 3)
            a *= self.stats['a_std'].view(1, 2).expand(a.size()).cuda()
            a += self.stats['a_mean'].view(1, 2).expand(a.size()).cuda()

        return a, None, None, None  # Returning a tuple of 4, for consistency with the stochastic policy


class ValueFunction(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.encoder = encoder(opt, 0, opt.ncond)
        self.hsize = opt.nfeature * self.opt.h_height * self.opt.h_width
        self.n_outputs = 1
        self.proj = nn.Linear(self.hsize, opt.n_hidden)

        self.fc = nn.Sequential(
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.Dropout(p=opt.dropout, inplace=True),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, self.n_outputs)
        )

    def forward(self, state_images, states, context=None, sample=True,
                normalize_inputs=False, normalize_outputs=False, n_samples=1):
        bsize = state_images.size(0)
        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.proj(h)
        h = self.fc(h).view(bsize, self.n_outputs)
        return h



# Mixture Density Network model
class PolicyMDN(nn.Module):
    def __init__(self, opt, n_mixture=10, npred=1):
        super(PolicyMDN, self).__init__()
        self.opt = opt
        self.npred = npred
        if not hasattr(opt, 'n_mixture'):
            self.opt.n_mixture = n_mixture
        self.encoder = encoder(opt, 0, opt.ncond)
        self.hsize = opt.nfeature * self.opt.h_height * self.opt.h_width
        self.n_outputs = self.npred*opt.n_actions
        self.fc = nn.Sequential(
            nn.Linear(self.hsize, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(),
            nn.Linear(opt.n_hidden, opt.n_hidden)
        )

        self.pi_net = nn.Linear(opt.n_hidden, opt.n_mixture)
        self.mu_net = nn.Linear(opt.n_hidden, opt.n_mixture*self.n_outputs)
        self.sigma_net = nn.Linear(opt.n_hidden, opt.n_mixture*self.n_outputs)

    def forward(self, state_images, states, sample=False, normalize_inputs=False, normalize_outputs=False):

        if normalize_inputs:
            # policy network is trained with states normalized by mean and standard dev.
            # this is to unnormalize the predictions at evaluation time.
            state_images = state_images.clone().float().div_(255.0)
            states -= self.stats['s_mean'].view(1, 4).expand(states.size())
            states /= self.stats['s_std'].view(1, 4).expand(states.size())
            state_images = state_images.cuda().unsqueeze(0)
            states = states.cuda().unsqueeze(0)

        bsize = state_images.size(0)
        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.fc(h)
        # get parameters of output distribution
        pi = F.softmax(self.pi_net(h).view(bsize, self.opt.n_mixture), dim=1)
        mu = self.mu_net(h).view(bsize, self.opt.n_mixture, self.n_outputs)
        sigma = F.softplus(self.sigma_net(h)).view(bsize, self.opt.n_mixture, self.n_outputs)
        if sample:
            # pick a mixture component (one for each element in minibatch)
            k = torch.multinomial(pi, 1)
            a = []
            for b in range(bsize):
                # sample from Gaussian associated with those components
                a.append(torch.randn(self.npred, self.opt.n_actions).cuda() *
                         sigma[b][k[b]].data.view(self.npred, self.opt.n_actions) +
                         mu[b][k[b]].data.view(self.npred, self.opt.n_actions))
            a = torch.stack(a).squeeze()
            a = a.view(bsize, self.npred, 2)
        else:
            a = None

        if normalize_outputs:
            a *= self.stats['a_std'].view(1, 1, 2).expand(a.size()).cuda()
            a += self.stats['a_mean'].view(1, 1, 2).expand(a.size()).cuda()

        return pi, mu, sigma, a

MODEL_DICT = {
    'fwd-cnn': FwdCNN,             # deterministic model
    'fwd-cnn-vae-fp': FwdCNN_VAE,  # stochastic model
    'fwd-obj': FwdObj,             # object based model
}
