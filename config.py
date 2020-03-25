from easydict import EasyDict as ed
config = ed()

# log
config.log_dir = './log'
config.log_interval = 100  # STEP

# data
config.train_data_root = './anime_colorization/data/train/'
config.val = True
config.val_data_root = './anime_colorization/data/val'
config.binary = False
config.orig_size = 512
config.train_size = 256  # or None

# model
config.apex = False
config.gpu = True
config.resume = False
config.resume_from_best = False
config.checkpoint_dir = './checkpoints'
config.dnet_slug = 'r18'
config.bn = True  # batch norm OR layer norm

# loss
config.loss_weight = [0.88, 0.79, 0.63, 0.51, 0.39, 1.07]
config.alpha = 0.999
config.beta = 0.001

# hyperparameters
config.epoch = 100
config.batch_size = 4
config.lr = 1e-4
config.save_interval = 1  # EPOCH
