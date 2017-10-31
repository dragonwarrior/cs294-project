import time
import math
import pickle
import numpy as np
import tensorflow as tf

def load_data(fname="data/hopper_data.pkl"):
    print("load data from %s" % (fname))
    train_data = pickle.load(open(fname, "rb"))
    return {"x": train_data["observations"].astype(np.float32), 
            "y": train_data["actions"].astype(np.float32),
            "mean": train_data["mean"],
            "std": train_data["std"]}


def simple_model(x, input_dim, output_dim):
    h1 = tf.contrib.layers.fully_connected(inputs=x, \
                                            num_outputs=100,\
                                            activation_fn=tf.nn.relu,\
                                            normalizer_fn=None)

    out = tf.contrib.layers.fully_connected(inputs=h1, \
                                            num_outputs=output_dim,\
                                            activation_fn=None)

    return out


def run_model(sess, model, xd, yd, epoch_num=1, \
                batch_size=10, is_training=True, \
                x_val=None, y_val=None, log_interval=1):

    print("run_model with epoch_num:%d, batch_size:%d" % (epoch_num, batch_size))

    sample_num = xd.shape[0]
    train_indicies = np.arange(sample_num)

    variables = [model['loss']]
    if is_training:
        variables.append(model['train_step'])

    validation_vars = None
    if x_val is not None and y_val is not None:
        validation_vars = [model['loss']]

    max_iters = epoch_num*int(math.ceil(sample_num/batch_size))
    batch_num = int(math.ceil(sample_num/batch_size))
    np.random.shuffle(train_indicies)
    for it in range(max_iters):
        i = it % batch_num 
        si = (i * batch_size) % sample_num
        idx = train_indicies[si: si + batch_size]

        # create a feed dictionary for this batch
        feed_dict = {model['x']: xd[idx, :],
                    model['y']: yd[idx, :],
                    model['is_training']: is_training}

        actual_batch_size = yd[idx].shape[0]

        train_loss = sess.run(variables, feed_dict=feed_dict)[0]
        if it % log_interval == 0 or it == max_iters - 1:
            loss_val = 0.0
            if validation_vars is not None:
                feed_dict = {model['x']: x_val, \
                            model['y']: y_val}

                loss_val = sess.run(validation_vars, feed_dict=feed_dict)[0]

            print("iter:%d, train_loss:%.5f, val_loss:%.5f" % (it, train_loss, loss_val))


def train(data, epoch_num=100, batch_size=100, log_interval=1):
    x_data = data['x']
    y_data = data['y']
    sample_num, input_dim = x_data.shape
    output_dim = y_data.shape[-1]

    num_training = int(sample_num * 0.6)
    num_validation = int(sample_num * 0.2)
    num_test = sample_num - num_training - num_validation

    indices = np.arange(sample_num)
    np.random.shuffle(indices)

    mask = indices[range(num_training)]
    x_train = x_data[mask]
    y_train = y_data[mask]

    mask = indices[range(num_training, num_training + num_validation)]
    x_val = x_data[mask]
    y_val = y_data[mask]

    mask = indices[range(num_training + num_validation, sample_num)]
    x_test = x_data[mask]
    y_test = y_data[mask]

    print("loaded train data:")
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)

    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)

    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)
    time.sleep(3)

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, input_dim], name="obs")
    y = tf.placeholder(tf.float32, shape=[None, output_dim], name="act")
    is_training = tf.placeholder(tf.bool)

    y_out = simple_model(x, input_dim, output_dim)

    #loss = tf.nn.l2_loss(y - y_out)
    loss = 0.5 * tf.reduce_sum(tf.square(y - y_out), axis=1)
    loss = tf.reduce_mean(loss)
    opt = tf.train.RMSPropOptimizer(learning_rate=1e-4, \
                                            decay=0.9, \
                                            momentum=0.5)
    train_step = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model = {'x': x, 'y': y, 'is_training': is_training, 
                 'loss': loss, 'train_step': train_step}
        run_model(sess, model, x_train, y_train, \
                    epoch_num, batch_size, True,
                    x_val=x_val, y_val=y_val, log_interval=log_interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default="data/hopper_data.pkl", 
                        help="train data file name, default to data/hopper_data.pkl")

    parser.add_argument('--epoch_num', type=int, default=1, 
                        help="num epochs to train, default to 1")

    parser.add_argument('--batch_size', type=int, default=100, 
                        help="num samples in one minibatch, default to 100")

    parser.add_argument('--log_interval', type=int, default=10, 
                        help="log result per rainning iters, default to 10")

    args = parser.parse_args()

    data = load_data(args.train_data)
    train(data, epoch_num=args.epoch_num, 
            batch_size=args.batch_size,
            log_interval=args.log_interval)
