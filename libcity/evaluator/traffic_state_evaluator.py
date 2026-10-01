import numpy as np
from libcity.evaluator.result_metrics import measured_metrics
import os
import json
import datetime
import pandas as pd
from libcity.utils import ensure_dir
from libcity.model import loss
from logging import getLogger
from libcity.evaluator.abstract_evaluator import AbstractEvaluator


class TrafficStateEvaluator(AbstractEvaluator):

    def __init__(self, config):
        self.metrics = config.get('metrics', ['MAE'])  # 评估指标, 是一个 list
        self.allowed_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE',
                                'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR']
        self.save_modes = config.get('save_mode', ['csv', 'json'])
        self.mode = config.get('evaluator_mode', 'single')  # or average
        self.config = config
        self.min_s = config.get('min_s', 1e-4)
        self.len_timeslots = 0
        self.result = {}  # 每一种指标的结果
        self.intermediate_result = {}  # 每一种指标每一个batch的结果
        self._check_config()
        self._logger = getLogger()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficStateEvaluator'.format(str(metric)))

    def collect(self, batch):
        y_true = batch['y_true'].detach().cpu().numpy()
        y_pred = batch['y_pred'].detach().cpu().numpy()
        if y_true.shape != y_pred.shape or y_true.ndim < 2:
            raise ValueError('Prediction and target shapes differ')
        if self.len_timeslots and self.len_timeslots != y_true.shape[1]:
            raise ValueError('Horizon differs across evaluation batches')
        self.len_timeslots = y_true.shape[1]
        self.intermediate_result.setdefault('truth', []).append(y_true)
        self.intermediate_result.setdefault('prediction', []).append(y_pred)


    def evaluate(self):
        truth = np.concatenate(self.intermediate_result['truth'])
        prediction = np.concatenate(self.intermediate_result['prediction'])
        for i in range(1, self.len_timeslots + 1):
            if self.mode.lower() == 'single':
                tr, pr = truth[:, i-1], prediction[:, i-1]
            elif self.mode.lower() == 'average':
                tr, pr = truth[:, :i], prediction[:, :i]
            else:
                raise ValueError('Unknown evaluator mode')
            full = measured_metrics(pr, tr)
            selected = np.isfinite(tr) & (np.abs(tr) >= self.min_s) & (tr != 0)
            masked = measured_metrics(pr[selected], tr[selected])
            for metric in self.metrics:
                self.result[metric + '@' + str(i)] = (masked[metric[7:]] if metric.startswith('masked_') else full[metric])
            for count in ['observed_count', 'total_count', 'mape_count']:
                self.result[count + '@' + str(i)] = full[count]
            self.result['masked_sample_count@' + str(i)] = masked['sample_count']
        return self.result


    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        self._logger.info('Note that you select the {} mode to evaluate!'.format(self.mode))
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:  # 使用时间戳
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is ' + json.dumps(self.result))
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.json'.format(filename)))

        dataframe = {}
        if 'csv' in self.save_modes:
            columns = self.metrics + ['observed_count', 'total_count', 'mape_count', 'masked_sample_count']
            for metric in columns:
                dataframe[metric] = []
            for i in range(1, self.len_timeslots + 1):
                for metric in columns:
                    dataframe[metric].append(self.result[metric + '@' + str(i)])
            dataframe = pd.DataFrame(dataframe, index=range(1, self.len_timeslots + 1))
            dataframe.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
            self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.csv'.format(filename)))
            self._logger.info("\n" + str(dataframe))
            self._logger.info("\n" + str(dataframe.mean()))
        return dataframe

    def clear(self):
        """
        清除之前收集到的 batch 的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        """
        self.len_timeslots = 0
        self.result = {}
        self.intermediate_result = {}
