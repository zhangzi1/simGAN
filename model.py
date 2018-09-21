from buffer import *
from data import *
from layers import *
from sample import *


def refiner(scope, input, reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scp:
        output = conv(input, 64, 3, 1, "conv1")
        output = res_block(output, 64, 3, 1, "block1")
        output = res_block(output, 64, 3, 1, "block2")
        output = res_block(output, 64, 3, 1, "block3")
        output = res_block(output, 64, 3, 1, "block4")
        output = conv(output, 1, 1, 1, "conv2")
        output = tf.nn.tanh(output)
        refiner_vars = tf.contrib.framework.get_variables(scp)
    return output, refiner_vars


def discriminator(scope, input, reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scp:
        output = conv(input, 96, 3, 2, scope="conv1")
        output = conv(output, 64, 3, 2, scope="conv2")
        output = max_pool(output, 3, 1)
        output = conv(output, 32, 3, 1, scope="conv3")
        output = conv(output, 32, 1, 1, scope="conv4")
        logits = conv(output, 2, 1, 1, scope="conv5")
        output = tf.nn.softmax(logits, name="softmax")
        discriminator_vars = tf.contrib.framework.get_variables(scp)
    return output, logits, discriminator_vars


# Eliminating gradient explosion
def minimize(optimizer, loss, vars, max_grad_norm):
    grads_and_vars = optimizer.compute_gradients(loss)
    new_grads_and_vars = []
    for i, (grad, var) in enumerate(grads_and_vars):
        if grad is not None and var in vars:
            new_grads_and_vars.append((tf.clip_by_norm(grad, max_grad_norm), var))
    return optimizer.apply_gradients(new_grads_and_vars)


class Model:
    def __init__(self):
        # Placeholder
        self.R_input = tf.placeholder(tf.float32, [None, None, None, 1])
        self.D_image = tf.placeholder(tf.float32, [None, None, None, 1])

        # [None,None,None,1]->[-1,35,55,1]
        self.r_input = tf.image.resize_images(self.R_input, [35, 55])
        self.d_image = tf.image.resize_images(self.D_image, [35, 55])

        # Network
        self.R_output, self.refiner_vars = refiner("Refiner", self.r_input)
        self.D_fake_output, self.D_fake_logits, self.discriminator_vars = discriminator("Discriminator", self.R_output)
        self.D_real_output, self.D_real_logits, _ = discriminator("Discriminator", self.d_image, True)

        # Refiner loss
        self.self_regulation_loss = tf.reduce_sum(tf.abs(self.R_output - self.r_input), [1, 2, 3],
                                                  name="self_regularization_loss", )
        self.refine_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.D_fake_logits,
                                                           labels=tf.ones_like(self.D_fake_logits, dtype=tf.int32)[:, :,
                                                                  :,
                                                                  0]),
            [1, 2])
        self.refiner_loss = tf.reduce_mean(0.5 * self.self_regulation_loss + self.refine_loss)

        # Discriminator loss
        self.discriminate_real_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.D_real_logits,
                                                           labels=tf.ones_like(self.D_real_logits, dtype=tf.int32)[:, :,
                                                                  :,
                                                                  0]),
            [1, 2])
        self.discriminate_fake_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.D_fake_logits,
                                                           labels=tf.zeros_like(self.D_fake_logits, dtype=tf.int32)[:,
                                                                  :, :,
                                                                  0]),
            [1, 2])
        self.discriminator_loss = tf.reduce_mean(self.discriminate_real_loss + self.discriminate_fake_loss)

        # Training step
        self.optimizer = tf.train.GradientDescentOptimizer(0.001)
        self.sf_step = minimize(self.optimizer, self.self_regulation_loss, self.refiner_vars, 50)
        self.refiner_step = minimize(self.optimizer, self.refiner_loss, self.refiner_vars, 50)
        self.discriminator_step = minimize(self.optimizer, self.discriminator_loss, self.discriminator_vars, 50)

        # Saver
        self.saver = tf.train.Saver()

        # Session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Summary
        tf.summary.scalar("Refiner Loss", self.refiner_loss)
        tf.summary.scalar("Discriminator Loss", self.discriminator_loss)
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./graphs", self.sess.graph)

        # Path setting
        self.data = Data("./data/syn/", "./data/real/")
        self.buffer = Buffer("./buffer/")
        self.sample = Sample("./samples/")
        self.syn_sample = self.data.syn_sample(1)
        self.real_sample = self.data.real_sample(1)
        self.syn_batch = self.data.syn_sample(16)

    def train_sr(self):
        if not os.path.exists("./logs/step1/"):
            print("[*] Training starts.")
            for i in range(1000):

                mini_batch = self.data.syn_sample(32)
                self.sess.run(self.sf_step, feed_dict={self.R_input: mini_batch})
                self.buffer.push(self.sess.run(self.R_output, feed_dict={self.R_input: mini_batch}))

                print((i + 1) / 10, "%", "SRL:",
                      self.sess.run(tf.reduce_mean(self.self_regulation_loss),
                                    feed_dict={self.R_input: self.syn_sample}))
                summary = self.sess.run(self.merged_summary, feed_dict={self.R_input: self.syn_sample})
                self.writer.add_summary(summary, global_step=i)

                if (i + 1) % 100 == 0:
                    self.sample.push(self.sess.run(self.R_output, feed_dict={self.R_input: self.syn_sample}))

            print("[*] Step 1 finished. ")
            self.saver.save(self.sess, "./logs/step1/")
        else:
            print("[*] Step 1 finished. ")
            self.saver.restore(self.sess, "./logs/step1/")

    def train_d(self):
        if not os.path.exists("./logs/step2/"):
            print("[*] Training starts.")
            for i in range(200):
                syn_batch = self.data.syn_sample(32)
                real_batch = self.data.real_sample(32)
                self.sess.run(self.discriminator_step, feed_dict={self.R_input: syn_batch, self.D_image: real_batch})
                print((i + 1) / 2, "%", "DL:",
                      self.sess.run(self.discriminator_loss,
                                    feed_dict={self.R_input: self.syn_sample, self.D_image: self.real_sample}))
                summary = self.sess.run(self.merged_summary,
                                        feed_dict={self.R_input: self.syn_sample, self.D_image: self.real_sample})
                self.writer.add_summary(summary, global_step=i)

            print("[*] Step 2 finished. ")
            self.saver.save(self.sess, "./logs/step2/")

        else:
            print("[*] Step 2 finished. ")
            self.saver.restore(self.sess, "./logs/step2/")

    def train(self):

        # Read log
        current_iteration = int(open("./logs/log.txt").read())

        # Stuff buffer
        if current_iteration == 0:
            stuff_batch = self.data.syn_sample(100)
            self.buffer.push(self.sess.run(self.R_output, feed_dict={self.R_input: stuff_batch}))

        # Read parameter
        if os.path.exists("./logs/step3/"):
            self.saver.restore(self.sess, "./logs/step3/")

        # Training
        print("[*] Training continues from iteration" + str(current_iteration))
        for i in range(1000):

            # Train refiner
            for j in range(2):
                mini_batch = self.data.syn_sample(32)
                self.sess.run(self.refiner_step, feed_dict={self.R_input: mini_batch})
                print((i + 1) / 100.0, "%", "RL:",
                      self.sess.run(self.refiner_loss, feed_dict={self.R_input: self.syn_sample}))

            # Mix
            new_syn_sample = self.data.syn_sample(16)
            new_refined_batch = self.sess.run(self.R_output, feed_dict={self.R_input: new_syn_sample})
            history_batch = self.buffer.sample(16)
            concat_batch = np.concatenate([new_refined_batch, history_batch], axis=0)

            # Train discriminator
            for k in range(1):
                real_batch = self.data.real_sample(32)
                self.sess.run(self.discriminator_step,
                              feed_dict={self.R_input: concat_batch, self.D_image: real_batch})
                print((i + 1) / 100.0, "%", "DL:",
                      self.sess.run(self.discriminator_loss,
                                    feed_dict={self.R_input: self.syn_sample, self.D_image: self.real_sample}))

            self.buffer.random_replace(new_refined_batch)

            # Summary
            summary = self.sess.run(self.merged_summary,
                                    feed_dict={self.R_input: self.syn_sample, self.D_image: self.real_sample})
            self.writer.add_summary(summary, global_step=i)

            # Sample
            sample_batch = self.sess.run(self.R_output, feed_dict={self.R_input: self.syn_batch})
            self.sample.push(concat(sample_batch))

        # Save parameter
        print("[*] Step 3 finished. ")
        self.saver.save(self.sess, "./logs/step3/")

        # Write log
        f = open("./logs/log.txt", "w")
        f.write(str(current_iteration + 1000))
        f.close()
