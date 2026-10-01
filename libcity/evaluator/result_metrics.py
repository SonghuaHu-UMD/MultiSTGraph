"""Evaluation from measured predictions, with explicit sample coverage."""
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score


def evaluation_arrays(cache, valid_sample_count=None):
    prediction, truth = cache['prediction'], cache['truth']
    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError('Expected matching [sample, horizon, node, feature] arrays')
    if valid_sample_count is None:
        if 'valid_sample_count' not in cache:
            raise ValueError('Legacy cache lacks valid_sample_count; rerun evaluation or supply its verified real sample count')
        raw = np.asarray(cache['valid_sample_count'])
        if raw.size != 1:
            raise ValueError('valid_sample_count must be one integer')
        valid_sample_count = raw.item()
    if (isinstance(valid_sample_count, (bool, np.bool_))
            or not isinstance(valid_sample_count, (int, np.integer))
            or not 0 < valid_sample_count <= len(prediction)):
        raise ValueError('Invalid valid_sample_count')
    return prediction[:valid_sample_count], truth[:valid_sample_count]


def measured_metrics(prediction, truth, threshold=None):
    prediction, truth = np.asarray(prediction, dtype=float), np.asarray(truth, dtype=float)
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and truth shapes differ')
    observed = np.isfinite(truth)
    if np.any(observed & ~np.isfinite(prediction)):
        raise ValueError('Nonfinite prediction for an observed target')
    selected = observed if threshold is None else observed & (truth > threshold)
    pr, tr = prediction[selected], truth[selected]
    count, observed_count = len(tr), int(observed.sum())
    result = {name: float('nan') for name in ['MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']}
    if count:
        error = pr - tr
        result.update(MAE=float(np.mean(np.abs(error))), MSE=float(np.mean(error ** 2)))
        result['RMSE'] = float(np.sqrt(result['MSE']))
        nonzero = tr != 0
        if np.any(nonzero):
            result['MAPE'] = float(np.mean(np.abs(error[nonzero] / tr[nonzero])))
        if count > 1 and np.var(tr) > 0:
            result.update(R2=float(r2_score(tr, pr)), EVAR=float(explained_variance_score(tr, pr)))
    result.update(sample_count=count, observed_count=observed_count,
                  total_count=int(truth.size), excluded_count=int(truth.size) - count,
                  mape_count=int(np.count_nonzero(tr)) if count else 0,
                  threshold=threshold)
    return result


def metric_values(prediction, truth):
    result = measured_metrics(prediction, truth)
    return [result[name] for name in ['MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']]


def horizon_metrics(frame, horizon, threshold=10):
    rows = frame[frame['ahead_step'] == horizon]
    result = measured_metrics(rows['prediction_t'], rows['truth_t'])
    subset = measured_metrics(rows['prediction_t'], rows['truth_t'], threshold=threshold)
    result.update({'subset_' + key: value for key, value in subset.items()})
    result['negative_predictions_clipped'] = int(rows.get('negative_prediction', np.zeros(len(rows))).sum())
    return result
