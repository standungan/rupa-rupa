import os

class configuration():
    
    #hardware
    device = "cuda"

    #dataset & dataloader
    root_train = "E:/DATASET/faces_CASIA_112x112/imgs"
    root_test  = None
    num_workers = 2
    batch_size = 32
    shuffle = True
    input_size = (112,112)

    #model
    backbone = 'mobile'
    num_layers = 50
    mode = 'ir_se'
    num_classes = len(os.listdir(root_train))

    #arcface
    embed_size = 32
    scale = 64.
    margin = 0.5

    #loss function
    # loss = 'cross_entropy'
    loss = 'focal_loss'

    #optimizer
    opt  = 'SGD'
    lr = 1e-3
    momentum = 0.9
    lr_step = 20
    weight_decay = 5e-4
    
    #training
    epochs = 60
    save_interval = 10000
    show_progress = 25
    checkpoint_path = 'checkpoint/'
    pretrained = True
    pretrained_path = 'checkpoint/ResNet50_ir_se_0.0_150000.pth' 
    log_dir = 'log/'
    log_filename = backbone + '_' + str(embed_size) + '.txt'

    #validation
    test_list = './test_pair_list.txt'
    state_path = 'state/'
    source_root = "E:/DATASET/data_uji/source"
    target_root = "E:/DATASET/data_uji/target"