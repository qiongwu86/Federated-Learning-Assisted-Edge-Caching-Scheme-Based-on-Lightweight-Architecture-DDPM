import numpy as np
import torch
from itertools import chain
from options import args_parser
from torch.utils.data import DataLoader
from dataset_processing import cache_efficiency, sampling_mobility,cache_efficiency3,idx_train3,request_delay2
from data_set import convert
from model_ddpm import LightweightUNet1D, GaussianMultinomialDiffusion
from model_ae import generator_data, AutoEncoder,train_autoencoder1
from train_fed import train_hfl
from env_communicate import Environ


if __name__ == '__main__':
    args = args_parser()
    in_out_dim = args.in_out_dim
    client_num = args.clients_num
    hidden_dim = 64
    num_step = args.num_step

    idx=0
    # gpu or cpu
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    sample, users_group_train, users_group_test, users_group_pre, user_request_num = sampling_mobility(args, args.clients_num)
    test_idx = []
    for i in range(client_num):
        test_idx.append(users_group_test[i])
    test_idx = list(chain.from_iterable(test_idx))
    user_dis = np.random.randint(0,500,client_num)
    env = Environ(client_num)
    env.new_random_game(user_dis)

    model = LightweightUNet1D()
    gl_dm = GaussianMultinomialDiffusion(num_numerical_features=in_out_dim,denoise_fn=model,num_timesteps=num_step,device=torch.device('cpu'))

    train_idx1 = idx_train3(user_request_num)
    gl_ae = AutoEncoder(input_dim=3952, hidden_dim=100, latent_dim=in_out_dim)
    Train_data1 = []
    for i in range(client_num):
        num = np.random.randint(30000,40000)
        train_idx = users_group_train[i][:num]
        train_data = convert(sample[train_idx], int(max(sample[:, 1])))
        train_data = torch.Tensor(train_data).float()
        Train_data1.append(train_data)
    cache_hit_ratio_500 = []
    cache_hit_ratio_100 = []
    client_epoch_time_all = []
    ae_am_delay = []
    ae_am_delay_100 = []
    v2i_rate = env.Compute_Performance_Train_mobility(client_num)
    train_pre = [x for sublist in users_group_pre.values() for x in sublist]
    pre_train = convert(sample[train_pre], int(max(sample[:, 1])))
    pre_train_data = torch.Tensor(pre_train).float()

    print(f"---------------------------- Pre-Training Start------------------------------")
    ae_dataloader = DataLoader(pre_train_data[:, 1:-3], batch_size=64, shuffle=True)
    train_ae = train_autoencoder1(gl_ae, args, ae_dataloader, device=torch.device('cpu'))

    while idx < args.epochs:
        print(f' | Global Training Round : {idx + 1}')
        gl_ae_w1,gl_dm_w1,client_epoch_time1 = train_hfl(args,train_idx1,Train_data1,gl_dm, train_ae)
        client_epoch_time_all.append(client_epoch_time1)
        train_ae.load_state_dict(gl_ae_w1)
        gl_dm.load_state_dict(gl_dm_w1)
        idx += 1
        generated_data = generator_data(gl_dm, 500)
        train_ae.eval()
        with torch.no_grad():
            generated_data_original_dim = train_ae.decoder(generated_data)
        generated_ratings = torch.sum(generated_data_original_dim, dim=0)
        if idx == args.epochs:
            generated_data = generator_data(gl_dm, 1000)
            train_ae.eval()
            with torch.no_grad():
                generated_data_original_dim = train_ae.decoder(generated_data)
            generated_ratings = torch.sum(generated_data_original_dim, dim=0)
            CACHE_HIT_RATIO_platoon , CACHE_HIT_RATIO_rsu, CACHE_HIT_RATIO_all = cache_efficiency(generated_ratings, test_idx, sample)
            request_number = sum(user_request_num)/len(user_request_num)
            v2i_rate_avg = sum(v2i_rate)/len(v2i_rate)
            ae_am_delay_100 = request_delay2(cache_hit_ratio_100, request_number, v2i_rate_avg)
            print('LDDPM_cache_efficiency:',CACHE_HIT_RATIO_all)

        if idx > args.epochs:
            break


