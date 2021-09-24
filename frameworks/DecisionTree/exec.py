import logging
import os
import tempfile
import pickle

import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute_array
from amlb.results import save_predictions
from amlb.utils import Timer

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** Decision Tree [sklearn v{sklearn.__version__}] ****\n")

    is_classification = config.type == 'classification'

    X_train, X_test = impute_array(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    estimator = DecisionTreeClassifier if is_classification else DecisionTreeRegressor
    predictor = estimator(random_state=config.seed, **config.framework_params)

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
                     target_is_encoded=is_classification)

    return dict(
        models_count=1,
        training_duration=training_duration,
        predict_duration=predict.duration,
        model_path=model_path,
    )
