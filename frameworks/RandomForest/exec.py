import logging
import os
import tempfile as tmp
import pickle
# import joblib

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer

log = logging.getLogger(os.path.basename(__file__))


def run(dataset, config):
    log.info(f"\n**** Random Forest [sklearn v{sklearn.__version__}] ****\n")

    is_classification = config.type == 'classification'

    encode = config.framework_params.get('_encode', True)
    X_train, X_test = dataset.train.X, dataset.test.X
    y_train, y_test = dataset.train.y, dataset.test.y

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    log.info("Running RandomForest with a maximum time of {}s on {} cores.".format(config.max_runtime_seconds, n_jobs))
    log.warning("We completely ignore the requirement to stay within the time limit.")
    log.warning("We completely ignore the advice to optimize towards metric: {}.".format(config.metric))

    if config.task_type == 'train':
        estimator = RandomForestClassifier if is_classification else RandomForestRegressor
        rf = estimator(n_jobs=n_jobs,
                       random_state=config.seed,
                       **training_params)

        with Timer() as training:
            rf.fit(X_train, y_train)
        training_duration = training.duration

        # save model
        model_path = tmp.mkdtemp()
        try:
            log.info('Saving model to %s', model_path)
            with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
                pickle.dump(rf, f)
            # joblib.dump(rf, os.path.join(model_path, 'model.joblib'))
        except:
            log.exception('Error saving model to %s', model_path)
            model_path = None

    elif config.task_type == 'predict':
        log.info('Loading model from %s', config.model_path)
        with open(os.path.join(config.model_path, 'model.pkl'), 'rb') as f:
            rf = pickle.load(f)
        # rf = joblib.load(os.path.join(model_path, 'model.joblib'))
        training_duration = 0.0
        model_path = None

    else:
        raise ValueError(f"Unknown task_type: {config.task_type}")

    with Timer() as predict:
        predictions = rf.predict(X_test)
    probabilities = rf.predict_proba(X_test) if is_classification else None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=encode,
                  models_count=len(rf),
                  training_duration=training_duration,
                  predict_duration=predict.duration,
                  model_path=model_path)


if __name__ == '__main__':
    call_run(run)
