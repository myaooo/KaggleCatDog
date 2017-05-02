import tensorflow as tf
import numpy as np

from cnn.data.preprocess import BATCH_SIZE
from cnn.convnet.utils import init_tf_environ, get_path
from cnn.data.preprocess import prep_data
from cnn.generate_submission import generate_submission
from cnn.model import build_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('models', '',
                           """A list of model number split by ','""")
tf.app.flags.DEFINE_string('names', '',
                           """A list of names of models""")
tf.app.flags.DEFINE_string('dataset', 'valid',
                           """which set of data to run""")
tf.app.flags.DEFINE_string('out', 'submissions/ensemble.csv',
                           """the path to the output""")


def ensemble_predict(data_generator, models):
    # n_model = len(models)
    predictions = []
    logits = []
    for model in models:
        logit, prediction = model.infer(model.sess, data_generator, batch_size=BATCH_SIZE)
        predictions.append(prediction)
        logits.append(logit)
    predictions = np.stack(predictions)
    prediction = np.mean(predictions, axis=0)
    return prediction


def cal_loss(predictions, labels):
    total_loss = 0
    for i, label in enumerate(labels):
        total_loss -= np.log(predictions[i, label])
    return total_loss/len(labels)


def cal_acc(predictions, labels):
    right_guess = 0.0
    for i, label in enumerate(labels):
        if label == np.argmax(predictions[i, :]):
            right_guess += 1.0
    return right_guess / len(labels)


def ensemble_eval(data_generator, models):

    predictions = ensemble_predict(data_generator, models)
    labels = data_generator.y
    ensemble_loss = cal_loss(predictions, labels)
    ensemble_acc = cal_acc(predictions, labels)
    print('ensemble loss:', ensemble_loss)
    print('ensemble accuracy:', ensemble_acc)
    return
    # losses = []
    # accs = []
    # for model in models:
    #     loss, acc, _ = model.eval(model.sess, data_generator, batch_size=BATCH_SIZE)
    #     losses.append(loss)
    #     accs.append(acc)


def main():
    init_tf_environ(gpu_num=1)
    all_data = prep_data(test=True)
    models = [int(num) for num in FLAGS.models.split(',')]
    names = FLAGS.names.split(',')
    dataset = FLAGS.dataset
    cnns = []
    for model, name in zip(models, names):
        cnn = build_model(model, name, *all_data[:2])
        cnn.restore()
        cnns.append(cnn)
    if dataset == 'test':
        predictions = ensemble_predict(all_data[2], cnns)
        generate_submission(predictions[:, 1], get_path(FLAGS.out))
        return
    elif dataset == 'train':
        d = 0
    else:
        d = 1
    ensemble_eval(all_data[d], cnns)


if __name__ == '__main__':
    main()
