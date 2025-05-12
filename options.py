import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    # DDPM 参数
    parser.add_argument('--epochs', type=int, default=20,
                        help="number of rounds of training")
    parser.add_argument('--lr', type=float, default=0.0006,
                        help='learning rate')
    parser.add_argument('--dm_lr_ep', type=int, default=30,
                        help="the number of local epochs: E")
    parser.add_argument('--num_step', type=int, default=50,
                        help="number of step: S")


    parser.add_argument('--AE_local_ep', type=int, default=15,
                        help="the number of local epochs: E")
    parser.add_argument('--pre_ae_ep', type=int, default=30,
                        help="the number of local epochs: E")
    parser.add_argument('--AE_lr', type=float, default=0.02,
                        help='learning rate')

    parser.add_argument('--in_out_dim', type=int, default=16,
                        help="in_out_dim")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--clients_num', type=int, default=5,
                        help="number of clients: K")
    parser.add_argument('--cachesize', type=list, default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                        help="size of cache: CS")

    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--decay', type=float, default=0.95,
                        help='Asy updata decay (default: 0.95)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Asy updata gamma (default: 0.5)')
    parser.add_argument('--update_decay', type=float, default=0.001,
                        help='Asy update_decay (default: 0.001)')
    # workspace arguments
    parser.add_argument('--clean_dataset', type=bool, default=True, help="clean\
                        the model/data_set or not")
    parser.add_argument('--clean_user', type=bool, default=True, help="clean\
                        the user/ or not")
    parser.add_argument('--clean_clients', type=bool, default=True, help="clean\
                        the model/clients or not")

    # workspace arguments density

    parser.add_argument('--clean_clients_density', type=bool, default=True, help="clean\
                        the model/clients or not")

    # data set
    parser.add_argument('--dataset', type=str, default='ml-1m', help="name of dataset")

    # other arguments
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
