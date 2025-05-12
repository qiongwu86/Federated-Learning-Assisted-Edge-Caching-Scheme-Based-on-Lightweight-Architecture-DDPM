import torch
def train_ddpm(args, train_loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.dm_lr_ep):
        total_loss = 0
        for ratings in train_loader:

            t = torch.randint(0, args.num_step, (ratings.size(0),))
            loss = model.compute_loss(ratings, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss, model

def aggregate_model_weights(global_model,client_weights):

    global_weights = client_weights[0][0]
    for key in global_weights.keys():
        for i in range(len(client_weights)):
            if i == 0:
                global_weights[key] = client_weights[0][1] * global_weights[key]
            else:
                w = client_weights[i][1]
                global_weights[key] = global_weights[key] + w * client_weights[i][0][key]

    for name, param in global_model.named_parameters():
        if name in global_weights:
            prev_param = global_weights[name]
            param.data.copy_(0.5 * prev_param + 0.5 * param.data)
    return global_model.state_dict()
def aggregate_model_weight(client_weights):
    """
    使用 FedAvg 聚合客户端的模型权重
    """
    total = 0
    for i in range(len(client_weights)):
        total += client_weights[i][1]
    # print('client_weights[0][1],total',client_weights[0][1],  total)
    global_weights = client_weights[0][0]
    for key in global_weights.keys():
        for i in range(len(client_weights)):
            if i == 0:
                global_weights[key] = (client_weights[0][1] / total) * global_weights[key]
            else:
                w = client_weights[i][1] / total
                global_weights[key] = global_weights[key] + w * client_weights[i][0][key]
    return global_weights

def aggregate_model_weights2(global_model,client_weights,w):
    """
    使用 FedAvg 聚合客户端的模型权重
    """
    global_weights = client_weights[0]
    for key in global_weights.keys():
        for i in range(len(client_weights)):
            if i == 0:
                global_weights[key] = w[i] * global_weights[key]
            else:
                global_weights[key] = global_weights[key] + w[i] * client_weights[i][key]
    return global_weights

def generator_data(global_model,num_samples):
    generated_data= global_model.sample(num_samples=num_samples)
    return generated_data
class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.Sigmoid()

        )
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
def train_autoencoder(autoencoder, args, dataloader, device):

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.AE_lr)
    loss_function = torch.nn.MSELoss()
    autoencoder.to(device)
    for epoch in range(args.AE_local_ep):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            output = autoencoder(data)
            loss = loss_function(output, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
    return autoencoder,avg_loss

def train_autoencoder1(autoencoder, args, dataloader, device):

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.AE_lr)
    loss_function = torch.nn.MSELoss()
    autoencoder.to(device)
    for epoch in range(args.pre_ae_ep):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            output = autoencoder(data)
            loss = loss_function(output, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        # if epoch%10==0 :
        print(f"------- Pre-training---Local Epoch {epoch}- Loss: {avg_loss:.4f}")
    return autoencoder
