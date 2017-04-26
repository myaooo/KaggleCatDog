import tensorflow as tf
import numpy as np

from cnn.data.preprocess import BATCH_SIZE

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('models', '',
                           """A list of model number split by ','""")
tf.app.flags.DEFINE_string('names', '',
                           """A list of names of models""")


def ensemble_predict(data_generator, models):
    # n_model = len(models)
    predictions = []
    for model in models:
        prediction = model.infer(model.sess, data_generator, batch_size=BATCH_SIZE)
        predictions.append(prediction)
    predictions = np.stack(predictions)
    prediction = np.mean(predictions, axis=0)
    return prediction


def ensemble_eval(data_generator, models):

    prediction = ensemble_predict(data_generator, models)

    losses = []
    accs = []
    for model in models:
        loss, acc, _ = model.eval(model.sess, data_generator, batch_size=BATCH_SIZE)
        losses.append(loss)
        accs.append(acc)

