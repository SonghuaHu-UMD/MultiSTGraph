import argparse
from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args

# "GRU", 'LSTM', 'RNN', 'Seq2Seq', 'FNN', 'AGCRN', 'MTGNN', 'STSGCN', 'STAGGCN', 'ASTGCN', 'STTN', 'STGCN', 'GWNET', 'TGCN','TGCLSTM', 'STG2Seq', 'GMAN', 'DCRNN'
model_list = ['STSGCN', 'STAGGCN', 'ASTGCN', 'STTN', 'STGCN', 'GWNET', 'TGCN', 'TGCLSTM', 'STG2Seq', 'GMAN', 'DCRNN']
if __name__ == '__main__':
    for model_name in model_list:
        print(model_name)
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='traffic_state_pred', help='the name of task')
        parser.add_argument('--model', type=str, default=model_name, help='the name of model')
        parser.add_argument('--dataset', type=str, default='SG_CTS_Hourly_Single_GP', help='the name of dataset')
        parser.add_argument('--config_file', type=str, default='config_user', help='the file name of config file')
        parser.add_argument('--saved_model', type=str2bool, default=True, help='whether save the trained model')
        parser.add_argument('--train', type=str2bool, default=True, help='whether re-train if the model is trained')
        parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        add_general_args(parser)
        args = parser.parse_args()
        dict_args = vars(args)
        other_args = {key: val for key, val in dict_args.items() if key not in
                      ['task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and val is not None}
        run_model(task=args.task, model_name=args.model, dataset_name=args.dataset, config_file=args.config_file,
                  saved_model=args.saved_model, train=args.train, other_args=other_args)
