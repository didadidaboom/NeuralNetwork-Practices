

class Hyperparameter:
    # ########################################################
    #                          Data
    # ########################################################
    device = "cpu" # cpu
    seed = 1234  #random seed
    in_features = 4 # input feature dimention
    out_dim = 2 # output class numbers

    # data root directory
    data_dir = "./data/"
    #original data path
    data_path = "./data/BankNote_Authentication.txt"
    testset_path = "./data/test.txt"
    devset_path = "./data/dev.txt"
    trainset_path = "./data/train.txt"

    # ########################################################
    #                   Model Structure
    # ########################################################
    layer_list = [in_features, 64, 128, 64, out_dim]
    # ########################################################
    #                     Experiment
    # ########################################################
    batch_size = 64
    init_lr = 1e-3
    epochs = 100
    verbos_step = 10
    save_step = 1000

HP = Hyperparameter()