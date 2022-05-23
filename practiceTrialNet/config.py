

class Hyperparameters:
    # ########################################################
    #                          Data
    # ########################################################
    device = "cpu"  #cuda
    data_root = "./data"
    cls_mapper_path = "./data/cls_mapper.json"
    train_data_root = "./data/Marcel-Train"
    test_data_root = "./data/Marcel-Test"

    metadata_train_path = "./data/train_hand_gesture.txt"
    metadata_eval_path = "./data/eval_hand_gesture.txt"
    metadata_test_path = "./data/test_hand_gesture.txt"

    classes_num = 6
    seed = 1234

    # ########################################################
    #                         Model
    # ########################################################
    data_channels = 3
    conv_kernel_size = 3
    fc_drop_prob = 0.3
    # ########################################################
    #                     Experiment
    # ########################################################
    batch_size = 2
    init_lr = 5e-4
    epochs = 100
    verbos_step = 250
    save_step = 500

HP = Hyperparameters()