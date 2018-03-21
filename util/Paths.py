tensorboard_logdir = '/z/home/mbanani/tensorboard_logs'

import os

pascal3d_root           = '/z/home/mbanani/datasets/pascal3d'

root_dir                = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
render4cnn_weights      =  os.path.join(root_dir, 'model_weights/r4cnn_12classes_stateDict.pkl')
clickhere_weights       =  os.path.join(root_dir, 'model_weights/chcnn_3classes_stateDict.pkl')
# ft_render4cnn_weights   =  os.path.join(root_dir, 'model_weights/ryan_render.npy')

kp_dict_chcnn_train         = os.path.join(root_dir, 'data/converted_CHCNN_pascalVehKP_train_kp_dict.npy')
kp_dict_chcnn_test          = os.path.join(root_dir, 'data/converted_CHCNN_pascalVehKP_test_kp_dict.npy')
kp_dict_chcnn_ftAtt_train   = os.path.join(root_dir, 'data/ftATT_CHCNN_pascalVehKP_train_kp_dict.npy')
kp_dict_chcnn_ftAtt_test    = os.path.join(root_dir, 'data/ftATT_CHCNN_pascalVehKP_test_kp_dict.npy')

r4cnnFT_dict_train            = os.path.join(root_dir, 'data/renderFT_dict_pascalVehKP_train_kp_dict.npy')
r4cnnFT_dict_test             = os.path.join(root_dir, 'data/renderFT_dict_pascalVehKP_test_kp_dict.npy')

r4cnn_dict_train            = os.path.join(root_dir, 'data/render_dict_pascalVehKP_train_kp_dict.npy')
r4cnn_dict_test             = os.path.join(root_dir, 'data/render_dict_pascalVehKP_test_kp_dict.npy')
