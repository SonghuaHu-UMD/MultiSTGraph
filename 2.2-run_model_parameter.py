import argparse
from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args

# ['multi', 'bidirection'], ['multi', 'unidirection'], ['multi', 'none']
model_list = ['MultiATGCN']
para_list = [5, 10, 20, 30, 40, 50]
if __name__ == '__main__':
    for model_name in model_list:
        for para in para_list:
            for random_seed in [0, 10, 100, 500, 5000]:
                parser = argparse.ArgumentParser()
                parser.add_argument('--task', type=str, default='traffic_state_pred', help='the name of task')
                parser.add_argument('--model', type=str, default=model_name, help='the name of model')
                parser.add_argument('--dataset', type=str, default='202001010601_DC_SG_CTractFIPS_Hourly_Single_GP',
                                    help='the name of dataset')
                parser.add_argument('--config_file', type=str, default='config_user',
                                    help='the file name of config file')
                parser.add_argument('--saved_model', type=str2bool, default=True, help='whether save the trained model')
                parser.add_argument('--train', type=str2bool, default=True,
                                    help='whether re-train if the model is trained')
                parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
                parser.add_argument('--seed', type=int, default=random_seed, help='random seed')
                parser.add_argument('--start_dim', type=int, default=0, help='start_dim')
                parser.add_argument('--end_dim', type=int, default=1, help='end_dim')
                parser.add_argument('--embed_dim', type=int, default=para, help='adjtype')
                add_general_args(parser)
                # args = parser.parse_args()
                args, unknown = parser.parse_known_args()
                dict_args = vars(args)
                other_args = {key: val for key, val in dict_args.items() if key not in
                              ['task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and val is not None}
                run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
                          config_file=args.config_file, saved_model=args.saved_model, train=args.train,
                          other_args=other_args)
