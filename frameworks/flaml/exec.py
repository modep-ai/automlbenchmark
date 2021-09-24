import logging
import os
import tempfile
import pickle

from flaml import AutoML, __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** FLAML [v{__version__}] ****\n")

    X_train, y_train = dataset.train.X, dataset.train.y.squeeze()
    X_test, y_test = dataset.test.X, dataset.test.y.squeeze()

    is_classification = config.type == 'classification'

    if config.task_type == 'train':
        time_budget = config.max_runtime_seconds
        n_jobs = config.framework_params.get('_n_jobs', config.cores)
        log.info("Running FLAML with {} number of cores".format(config.cores))
        aml = AutoML()

        # Mapping of benchmark metrics to flaml metrics
        metrics_mapping = dict(
            acc='accuracy',
            auc='roc_auc',
            f1='f1',
            logloss='log_loss',
            mae='mae',
            mse='mse',
            rmse='rmse',
            r2='r2',
        )
        perf_metric = metrics_mapping[
            config.metric] if config.metric in metrics_mapping else 'auto'
        if perf_metric is None:
            log.warning("Performance metric %s not supported.", config.metric)

        training_params = {k: v for k, v in config.framework_params.items()
                           if not k.startswith('_')}

        log_dir = output_subdir("logs", config)
        flaml_log_file_name = os.path.join(log_dir, "flaml.log")

        with Timer() as training:
            aml.fit(X_train, y_train,
                    metric=perf_metric,
                    task=config.type,
                    n_jobs=n_jobs,
                    log_file_name= flaml_log_file_name,
                    time_budget=time_budget, **training_params)
        training_duration = training.duration

        # save model
        model_path = tempfile.mkdtemp()
        try:
            log.info('Saving model to %s', model_path)
            with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
                pickle.dump(aml, f)
        except:
            log.exception('Error saving model to %s', model_path)
            model_path = None

    elif config.task_type == 'predict':
        log.info('Loading model from %s', config.model_path)
        with open(os.path.join(config.model_path, 'model.pkl'), 'rb') as f:
            aml = pickle.load(f)
        training_duration = 0.0
        model_path = None

    else:
        raise ValueError(f"Unknown task_type: {config.task_type}")
    
    with Timer() as predict:
        predictions = aml.predict(X_test)
    probabilities = aml.predict_proba(X_test) if is_classification else None
    labels = aml.classes_ if is_classification else None
    return result(  
                    output_file=config.output_predictions_file,
                    probabilities=probabilities,
                    predictions=predictions,
                    truth=y_test,
                    models_count=len(aml.config_history),
                    training_duration=training_duration,
                    predict_duration=predict.duration,
                    probabilities_labels=labels,
                    model_path=model_path,
                )


if __name__ == '__main__':
    call_run(run)
