import logging
import os
import tempfile
import pickle

from sklearn.dummy import DummyClassifier, DummyRegressor

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions
from amlb.utils import Timer

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Constant predictor (sklearn dummy) ****\n")

    is_classification = config.type == 'classification'
    predictor = DummyClassifier(strategy='prior') if is_classification else DummyRegressor(strategy='median')

    encode = config.framework_params.get('_encode', False)
    X_train = dataset.train.X_enc if encode else dataset.train.X
    y_train = dataset.train.y_enc if encode else dataset.train.y
    X_test = dataset.test.X_enc if encode else dataset.test.X
    y_test = dataset.test.y_enc if encode else dataset.test.y

    if config.task_type == 'train':
        with Timer() as training:
            predictor.fit(X_train, y_train)
        training_duration = training.duration

        # save model
        model_path = tempfile.mkdtemp()
        try:
            log.info('Saving model to %s', model_path)
            with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
                pickle.dump(predictor, f)
        except:
            log.exception('Error saving model to %s', model_path)
            model_path = None

    elif config.task_type == 'predict':
        log.info('Loading model from %s', config.model_path)
        with open(os.path.join(config.model_path, 'model.pkl'), 'rb') as f:
            predictor = pickle.load(f)
        training_duration = 0.0
        model_path = None

    else:
        raise ValueError(f"Unknown task_type: {config.task_type}")

    with Timer() as predict:
        predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test,
                     target_is_encoded=encode)

    return dict(
        models_count=1,
        training_duration=training_duration,
        predict_duration=predict.duration,
        model_path=model_path,
    )
