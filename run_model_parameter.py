import argparse
from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args

model_list = ['MultiATGCN']
# para_list = [[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [1, 2, 1], [1, 3, 1],
#              [2, 2, 1], [2, 3, 1], [3, 3, 1], [1, 1, 0], [1, 1, 2], [1, 1, 3]]
# para_list = [['od', 'bidirection'], ['od', 'unidirection'], ['od', 'none'], ['dist', 'none'], ['cosine', 'none'],
#              ['identity', 'none'], ['multi', 'bidirection']]
# para_list = [[True, True, True, True], [True, True, False, False], [False, True, False, False],
#              [False, True, False, True], [False, False, False, False]]
# para_list = [True, False]
# para_list= [16, 32, 64, 72]
# para_list = [1, 5, 10, 20, 30, 50]
para_list = [1, 2, 3]
# para_list = [False]
if __name__ == '__main__':
    for model_name in model_list:
        for para in para_list:
            for random_seed in [0, 10, 100, 1000]:
                for dataset in ['201901010601_BM_SG_CTractFIPS_Hourly_Single_GP']:
                    print(para)
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--task', type=str, default='traffic_state_pred', help='the name of task')
                    parser.add_argument('--model', type=str, default=model_name, help='the name of model')
                    parser.add_argument('--dataset', type=str, default=dataset, help='the name of dataset')
                    parser.add_argument('--config_file', type=str, default='config_user', help='config file')
                    parser.add_argument('--saved_model', type=str2bool, default=True, help='saved_model')
                    parser.add_argument('--train', type=str2bool, default=True,
                                        help='whether re-train if the model is trained')
                    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
                    parser.add_argument('--seed', type=int, default=random_seed, help='random seed')
                    parser.add_argument('--start_dim', type=int, default=0, help='start_dim')
                    parser.add_argument('--end_dim', type=int, default=1, help='end_dim')
                    # parser.add_argument('--embed_dim_node', type=int, default=para, help='embed_dim_node')
                    parser.add_argument('--cheb_order', type=int, default=para, help='cheb_order')
                    # parser.add_argument('--rnn_units', type=int, default=para, help='rnn_units')
                    # parser.add_argument('--adjtype', type=str, default=para[0], help='adjtype')
                    # parser.add_argument('--adpadj', type=str, default=para[1], help='adpadj')
                    # parser.add_argument('--node_specific_off', type=bool, default=para, help='node_specific_off')
                    # parser.add_argument('--gcn_off', type=bool, default=para, help='gcn_off')
                    # parser.add_argument('--fnn_off', type=bool, default=para, help='fnn_off')
                    # parser.add_argument('--load_dynamic', type=bool, default=para[0], help='load_dynamic')
                    # parser.add_argument('--add_time_in_day', type=bool, default=para[1], help='add_time_in_day')
                    # parser.add_argument('--add_day_in_week', type=bool, default=para[2], help='add_day_in_week')
                    # parser.add_argument('--add_static', type=bool, default=para[3], help='add_static')
                    # parser.add_argument('--len_closeness', type=int, default=para[0], help='len_closeness')
                    # parser.add_argument('--len_period', type=int, default=para[1], help='len_period')
                    # parser.add_argument('--len_trend', type=int, default=para[2], help='len_trend')
                    add_general_args(parser)
                    # args = parser.parse_args()
                    args, unknown = parser.parse_known_args()
                    dict_args = vars(args)
                    other_args = {key: val for key, val in dict_args.items() if key not in
                                  ['task', 'model', 'dataset', 'config_file', 'saved_model',
                                   'train'] and val is not None}
                    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
                              config_file=args.config_file, saved_model=args.saved_model, train=args.train,
                              other_args=other_args)
