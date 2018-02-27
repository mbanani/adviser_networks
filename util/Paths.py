tensorboard_logdir = '/z/home/mbanani/tensorboard_logs'

import os

pascal3d_root           = '/z/home/mbanani/datasets/pascal3d'

root_dir                = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
render4cnn_weights      =  os.path.join(root_dir, 'model_weights/r4cnn_12classes_stateDict.pkl')
clickhere_weights       =  os.path.join(root_dir, 'model_weights/chcnn_3classes_stateDict.pkl')
# ft_render4cnn_weights   =  os.path.join(root_dir, 'model_weights/ryan_render.npy')
