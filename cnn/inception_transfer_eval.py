from cnn.data.preprocess import prep_data
from cnn.generate_submission import convnet_submission
import tensorflow as tf
from cnn.inception.inception_model import inference
from cnn.inception.dataset import CatDogData
from cnn.inception import image_processing
from cnn.convnet.utils import get_path, init_tf_environ

num_epochs = 10
BATCH_SIZE = 50
EVAL_FREQUENCY = 1

FLAGS = tf.app.flags.FLAGS


def infer(dataset):
    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    images, _ = image_processing.distorted_inputs(
        dataset,
        num_preprocess_threads=num_preprocess_threads)
    logits, _ = inference(images, 2, False, restore_logits=False, scope="INFER")
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    infer_logits = sess.run(logits)
    print(infer_logits.shape)
    # softmax = infer_logits


def main():
    init_tf_environ(gpu_num=0)
    all_data = prep_data(test=True)
    # model = build_model(*all_data[:4])
    # model.restore()
    # convnet_submission(model, all_data[4], get_path('submissions/inception/submission.csv'))
    # model.(BATCH_SIZE, 1, EVAL_FREQUENCY)
    # model.save()
    FLAGS.subset = 'validation'
    dataset = CatDogData(subset=FLAGS.subset)
    assert dataset.data_files()
    infer(dataset)

if __name__ == '__main__':
    main()
