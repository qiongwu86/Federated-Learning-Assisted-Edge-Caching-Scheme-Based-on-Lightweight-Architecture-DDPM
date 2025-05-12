import copy
import torch
import time
from torch.utils.data import DataLoader
from model_ae import train_autoencoder, train_ddpm, aggregate_model_weight, aggregate_model_weights


def train_hfl(args,platoon_train_idx,train_data,gl_dm, gl_ae):
    client_epoch_time=[]
    ae_weights = []
    dm_weights = []
    for i in range(args.clients_num):
        start_time = time.time()
        ae_dataloader = DataLoader(train_data[i][:, 1:-3], batch_size=args.batch_size, shuffle=True)
        ae_model = copy.deepcopy(gl_ae)
        train_ae, ae_avg_loss = train_autoencoder(ae_model, args, ae_dataloader, device=torch.device('cpu'))
        agg_ae_w = train_ae.state_dict()
        train_ae.eval()
        with torch.no_grad():
            dim_data = train_ae.encode(train_data[i][:, 1:-3])
        tra_dm_load = DataLoader(dim_data, batch_size=args.batch_size, shuffle=True)
        dm_model = copy.deepcopy(gl_dm)
        dm_avg_loss, train_dm = train_ddpm(args,tra_dm_load, dm_model)
        train_dm_w = train_dm.state_dict()
        agg_ae_w = [agg_ae_w, platoon_train_idx[i]]
        train_dm_w = [train_dm_w, platoon_train_idx[i]]
        ae_weights.append(agg_ae_w)
        dm_weights.append(train_dm_w)
        client_epoch_time.append(time.time() - start_time)
        print(f"- User {i+1} ---- DDPM Training ----- Local Epoch {args.dm_lr_ep}- Loss: {dm_avg_loss:.4f}")
    agg_ae_w = aggregate_model_weight(ae_weights)
    agg_dm_w = aggregate_model_weight(dm_weights)
    return agg_ae_w,agg_dm_w, max(client_epoch_time)


