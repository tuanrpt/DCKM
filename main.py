import os
import tensorflow as tf
import time
import sys
import utils
import numpy as np
from sklearn import metrics as mt


class DCKM:
    def __init__(self):
        # Initial settings
        self.running_mode = 1  # 0: test, 1: train

        self.dataset_name = '6_projects'  # 6_projects' or 'NDSS18'
        self.data_size = 'full_dataset'  # 'windows_only' or 'ubuntu_only' or 'full_dataset'

        self.datetime = utils._DATETIME
        tf.set_random_seed(utils._RANDOM_SEED)
        np.random.seed(utils._RANDOM_SEED)
        self.logging_path = utils._LOG

        # Training settings
        self.batch_size = 64
        self.num_train_steps = 100
        self.display_step = 1
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.max_gradient_norm = 1.0
        self.learning_rate = 0.0005
        self.optimizer = 'adam'

        # Embeddings
        self.vocab_assembly_size = 256
        self.embedding_dimension = 100  # 6_projects: 100, NDSS18: 64

        # Bidirectional RNN
        self.hidden_size = 128  # 6_projects: 128, NDSS18: 256

        # Kernel Machine
        self.num_random_features = 512
        self.gamma_init = 0.5
        self.train_gamma = False
        self.lamda_l2 = 0.01

        utils.save_all_params(self)

    def _create_input(self):
        with tf.name_scope("input"):
            self.X_opcode = tf.placeholder(tf.float32, [None, self.time_steps, self.vocab_opcode_size], name='x_opcode_input')
            self.X_assembly = tf.placeholder(tf.float32, [None, self.time_steps, self.vocab_assembly_size], name='x_assemply_input')
            self.Y = tf.placeholder(tf.int32, [None], name='true_label')
            self.sequence_length = tf.placeholder(tf.int32, [None], name='seq_length')

    def _create_embedding(self):
        with tf.name_scope("embedding"):
            self.w_opcode = tf.Variable(tf.truncated_normal([self.vocab_opcode_size, self.embedding_dimension], stddev=0.05),
                                        name='w_opcode')

            self.w_assembly = tf.Variable(tf.truncated_normal([self.vocab_assembly_size, self.embedding_dimension], stddev=0.05),
                                          name='w_assembly')

            # (batch_size, time_steps, vocab_size) x (vocab_size, embedding_dimension) = (batch_size, time_steps, embedding_dimension)
            self.embed_opcode = tf.tensordot(self.X_opcode, self.w_opcode, axes=((2,), (0,)))
            self.embed_assembly = tf.tensordot(self.X_assembly, self.w_assembly, axes=((2,), (0,)))
            self.rnn_input = tf.concat([self.embed_opcode, self.embed_assembly], axis=2)  # (batch_size, time_steps, 2*embedding_dimension)

    def _create_bi_rnn(self):
        with tf.name_scope("bi-rnn"):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(cell, cell, self.rnn_input, dtype=tf.float32,
                                                                        sequence_length=self.sequence_length)
            self.g_h_concatination = tf.concat([self.states[0], self.states[1]], 1)  # (batch_size, 2*self.num_hidden)

    def _create_kernel_machine(self):
        with tf.name_scope("oc-svm"):
            self.input_of_rf = self.g_h_concatination

            log_gamma = tf.get_variable(name='log_gamma', shape=[1],
                                        initializer=tf.constant_initializer(
                                            np.log(self.gamma_init)),
                                        trainable=self.train_gamma)  # (1,)
            e = tf.get_variable(name="unit_noise", shape=[self.input_of_rf.get_shape()[1], self.num_random_features],
                                initializer=tf.random_normal_initializer(), trainable=False)
            omega = tf.multiply(tf.exp(log_gamma), e, name='omega')  # (2*128, n_random_features) = e.shape
            omega_x = tf.matmul(self.input_of_rf,
                                omega)  # (batch_size, 2*128) x (2*128, n_random_features) = (batch_size, n_random_features)
            phi_x_tilde = tf.concat([tf.cos(omega_x), tf.sin(omega_x)], axis=1,
                                    name='phi_x_tilde')  # (batch_size, 2*n_random_features)

            self.w_rf = tf.Variable(tf.truncated_normal((2 * self.num_random_features, 1), stddev=0.05),
                                    name='w_rf')  # (2*n_random_features, 1)

            self.rho2 = tf.Variable(0.0, dtype=tf.float32, name='rho2')
            self.w_phi_minus_rho2 = tf.matmul(phi_x_tilde, self.w_rf) - self.rho2  # (batch_size, 1)
            self.l2_regularization = self.lamda_l2 * tf.reduce_sum(tf.square(self.w_rf))

    def _create_loss_for_finding_2_parallel_hyperplanes(self):
        with tf.name_scope("loss_svm"):
            self.Y_svm = self.Y  # self.Y is belong to 1 (vul), 0 (non-vul)
            self.Y_svm = tf.subtract(tf.cast(self.Y_svm, tf.float32), 0.5)
            self.Y_svm = tf.reshape(tf.sign(self.Y_svm), shape=(self.batch_size, 1)) # self.Y_svm is belong to 1 (vul), -1 (non-vul)

            self.loss = tf.reduce_mean(
                tf.maximum(0.0, (1.0 + self.Y_svm)/2 - self.w_phi_minus_rho2 * self.Y_svm)) + self.l2_regularization

    def _create_optimizer(self):
        with tf.name_scope("train"):
            parameters = tf.trainable_variables()
            gradients = tf.gradients(self.loss, parameters)
            clipped_gradients, self.global_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'grad':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            self.grads = optimizer.compute_gradients(self.loss)
            self.training_op = optimizer.apply_gradients(zip(clipped_gradients, parameters),
                                                         global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope("visualize"):
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name + '/values', var)
            for grad, var in self.grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)

        self.summary_op = tf.summary.merge_all()

    def _create_logging_files(self):
        self.graph_path = os.path.join('graphs', self.data_size, self.datetime)
        self.checkpoint_path = os.path.join('saved-model', self.data_size, self.datetime)
        utils.make_dir(self.checkpoint_path)

    def build_model(self):
        self._create_input()
        self._create_embedding()
        self._create_bi_rnn()
        self._create_kernel_machine()
        self._create_loss_for_finding_2_parallel_hyperplanes()
        self._create_optimizer()
        self._create_summaries()
        self._create_logging_files()

    def compute_score(self, y_true, y_pred):
        accuracy_score = mt.accuracy_score(y_true=y_true, y_pred=y_pred)
        pre_score = mt.precision_score(y_true=y_true, y_pred=y_pred)
        f1_score = mt.f1_score(y_true=y_true, y_pred=y_pred)
        recall_score = mt.recall_score(y_true=y_true, y_pred=y_pred)
        auc_score = mt.roc_auc_score(y_true=y_true, y_score=y_pred)
        return accuracy_score, pre_score, f1_score, recall_score, auc_score

    def compute_cost_sensitive_loss_and_y_pred(self, current_predict, y_true, all_test_data=False):
        tmp_b = current_predict[current_predict > 0.0]
        tmp_c = y_true[current_predict > 0.0]

        d = tmp_b[tmp_b < 1.0]
        e = tmp_c[tmp_b < 1.0]
        # print("number of data points in strip: %d" % len(d))

        f = np.concatenate((np.expand_dims(d, axis=0), np.expand_dims(e, axis=0)), axis=0)
        '''
        sorted_data_in_strip: shape (2, m) where m is the number of data point in the strip and is sorted in ascending order
            first row: w*phi_tilde(x) - rho2
            second row: value of y_true
        '''
        sorted_data_in_strip = f[:, f[0].argsort()]

        # if there is one or no any data point in strip --> coefficient_a of optimal hyperplane will be set to 0.5
        if sorted_data_in_strip.shape[1] <= 1:
            y_pred_with_hyperplane_0 = np.sign(
                np.sign(current_predict) + 1.0)  # transform from {-1,1} to {0,1}

            pred_with_optimal_hyperplane = current_predict - 0.5
            y_pred_with_optimal_hyperplane = np.sign(
                np.sign(pred_with_optimal_hyperplane) + 1.0)  # transform from {-1,1} to {0,1}

            pred_with_hyperplane_1 = current_predict - 1.0
            y_pred_with_hyperplane_1 = np.sign(
                np.sign(pred_with_hyperplane_1) + 1.0)  # transform from {-1,1} to {0,1}

            return 0.0, y_pred_with_hyperplane_0, y_pred_with_optimal_hyperplane, y_pred_with_hyperplane_1, 0

        '''
            pairs_y_predict_y_true: shape (m-1, 2, m)
            vectorization all possible cases of hyperplanes. In this case, we'll consider all (m-1) hyperplanes
        '''
        pairs_y_predict_y_true = np.ones((sorted_data_in_strip.shape[1] - 1, 2, sorted_data_in_strip.shape[1])) + 1.0
        for id, y_pred_y_true in enumerate(pairs_y_predict_y_true):
            pairs_y_predict_y_true[id][0][:(id + 1)] = [0] * (id + 1)
            pairs_y_predict_y_true[id][1] = sorted_data_in_strip[1]

        m_tn_fn_fp_tp = np.sum(pairs_y_predict_y_true, axis=1)  # shape (m-1, m) where m > 1

        n_vul = int(np.sum(y_true))
        n_non_vul = len(y_true) - n_vul
        lamda = n_non_vul / n_vul

        cost_sensitive = []
        for tn_fn_fp_tp in m_tn_fn_fp_tp:
            count_tn_fn_fp_tp = utils.Counter(tn_fn_fp_tp)
            tn = count_tn_fn_fp_tp[0]  # the number of true negatives
            fn = count_tn_fn_fp_tp[1]  # the number of false negatives
            fp = count_tn_fn_fp_tp[2]  # the number of false positives
            tp = count_tn_fn_fp_tp[3]  # the number of true positives

            if fp + tn != 0 and fn + tp != 0:
                FPR = fp / (fp + tn)
                FNR = fn / (fn + tp)
            else:
                if fp + tn == 0 and fn + tp != 0:  # can not compute FPR
                    FPR = 0.0
                    FNR = fn / (fn + tp)
                elif fn + tp == 0 and fp + tn != 0:  # can not compute FNR
                    FPR = fp / (fp + tn)
                    FNR = 0.0
                else:
                    FPR = 0.0
                    FNR = 0.0

            cost_sensitive.append(lamda * FNR + FPR)

            # if (lamda * FNR + FPR) > 0.0:
            #     print(str(lamda * FNR + FPR) + '\n')
            #     cost_sensitive.append(lamda * FNR + FPR)

        min_cost_index = int(np.argmin(cost_sensitive))
        min_cost_value = cost_sensitive[min_cost_index]
        '''
        coefficient_ a is the coefficient 'a' of optimal hyperplane formula: w_phi_minus_rho2 - a = 0
        '''
        coefficient_a = (sorted_data_in_strip[0][min_cost_index] + sorted_data_in_strip[0][
            min_cost_index + 1]) / 2.0

        y_pred_with_hyperplane_0 = np.sign(
            np.sign(current_predict) + 1.0)  # transform from {-1,1} to {0,1}

        pred_with_optimal_hyperplane = current_predict - coefficient_a
        y_pred_with_optimal_hyperplane = np.sign(
            np.sign(pred_with_optimal_hyperplane) + 1.0)  # transform from {-1,1} to {0,1}

        pred_with_hyperplane_1 = current_predict - 1.0
        y_pred_with_hyperplane_1 = np.sign(
            np.sign(pred_with_hyperplane_1) + 1.0)  # transform from {-1,1} to {0,1}

        return min_cost_value, y_pred_with_hyperplane_0, y_pred_with_optimal_hyperplane, y_pred_with_hyperplane_1, \
               sorted_data_in_strip.shape[1]

    def train(self, x_train_opcode, x_train_assembly, x_train_seq_len, y_train,
              x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid):

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=utils.get_default_config()) as sess:
            writer = tf.summary.FileWriter(self.graph_path, sess.graph)

            check_point = tf.train.get_checkpoint_state(self.checkpoint_path)
            if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
                message = "Load model parameters from %s\n" % check_point.model_checkpoint_path
                utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)
                saver.restore(sess, check_point.model_checkpoint_path)
            else:
                message = "Create the model with fresh parameters\n"
                utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)
                sess.run(tf.global_variables_initializer())

            training_set = x_train_opcode.shape[0] - x_train_opcode.shape[0] % self.batch_size
            training_batches = utils.make_batches(training_set, self.batch_size)

            step_loss = 0.0  # the loss per epoch on average
            step_time = 0.0
            initial_step = self.global_step.eval()

            for step in range(initial_step, initial_step + self.num_train_steps):
                loss_per_batch = 0.0
                start_time = time.time()
                step_predict_train = np.array([])

                for batch_idx, (batch_start, batch_end) in enumerate(training_batches):
                    batch_x_opcode = utils.convert_list_sparse_to_dense(x_train_opcode[batch_start:batch_end])
                    batch_x_assembly = utils.convert_list_sparse_to_dense(x_train_assembly[batch_start:batch_end])
                    batch_sequence_length = x_train_seq_len[batch_start:batch_end]
                    batch_y = y_train[batch_start:batch_end]

                    train_feed_dict = {
                        self.X_opcode: batch_x_opcode,
                        self.X_assembly: batch_x_assembly,
                        self.Y: batch_y,
                        self.sequence_length: batch_sequence_length,
                    }
                    _, summary, batch_loss, train_batch_y_pred = sess.run(
                        [self.training_op, self.summary_op, self.loss, self.w_phi_minus_rho2],
                        feed_dict=train_feed_dict)

                    if (batch_idx + 1) % (len(training_batches) // 10) == 0:
                        writer.add_summary(summary, global_step=step)

                    loss_per_batch += batch_loss / len(training_batches)
                    step_predict_train = np.append(step_predict_train, train_batch_y_pred)

                    sys.stdout.write("\rProcessed %.2f%% of mini-batches" % (((batch_idx + 1) / len(training_batches)) * 100))
                    sys.stdout.flush()

                '''
                now we had full (w*phi_tilde(x) - rho2) in 'step_predict_train' after getting rid of the for loop above
                '''

                step_time += (time.time() - start_time) / self.display_step
                step_loss += loss_per_batch / self.display_step

                if (step + 1) % 10 == 0:
                    # save checkpoint
                    checkpoint_path = os.path.join(self.checkpoint_path, "rnn_classifier_" + self.data_size + ".ckpt")
                    saver.save(sess, checkpoint_path, global_step=step)

                if (step + 1) % self.display_step == 0:
                    print("\n")
                    message = "global step %d/%d step-time %.2fs average total loss %.5f\n" % (
                        step, self.num_train_steps - 1, step_time, step_loss)
                    utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)

                    # run evaluation and print the total loss
                    dev_set = x_valid_opcode.shape[0] - x_valid_opcode.shape[0] % self.batch_size
                    dev_batches = utils.make_batches(dev_set, self.batch_size)

                    average_dev_loss = 0.0
                    valid_full_pred = np.array([])
                    for batch_idx, (batch_start, batch_end) in enumerate(dev_batches):
                        valid_x_opcode = utils.convert_list_sparse_to_dense(x_valid_opcode[batch_start:batch_end])
                        valid_x_assembly = utils.convert_list_sparse_to_dense(x_valid_assembly[batch_start:batch_end])
                        valid_y = y_valid[batch_start:batch_end]
                        valid_seq_len = x_valid_seq_len[batch_start:batch_end]

                        valid_feed_dict = {
                            self.X_opcode: valid_x_opcode,
                            self.X_assembly: valid_x_assembly,
                            self.Y: valid_y,
                            self.sequence_length: valid_seq_len,
                        }
                        batch_dev_loss, valid_batch_pred = sess.run([self.loss, self.w_phi_minus_rho2],
                                                                      feed_dict=valid_feed_dict)
                        valid_full_pred = np.append(valid_full_pred, valid_batch_pred)
                        average_dev_loss += batch_dev_loss / len(dev_batches)

                    '''
                    now we have full (w*phi_tilde(x) - rho2) in the 'valid_full_pred' variable
                    '''
                    pred_train_and_valid_set = np.concatenate((step_predict_train, valid_full_pred), axis=0)
                    y_true_train_valid = np.concatenate((y_train[:training_set], y_valid[:dev_set]), axis=0)

                    train_val_min_cost_value, train_val_y_pred_0, train_val_y_pred_with_optimal_hyperplane, train_val_y_pred_1, train_val_n_data_in_strip = self.compute_cost_sensitive_loss_and_y_pred(
                        pred_train_and_valid_set, y_true_train_valid)

                    train_y_pred_0 = train_val_y_pred_0[:training_set]
                    valid_y_pred_0 = train_val_y_pred_0[training_set:]
                    step_train_acc_0, step_train_pre_0, step_train_f1_0, step_train_rec_0, step_train_auc_0 = self.compute_score(
                        y_true=y_train[:training_set], y_pred=train_y_pred_0)

                    train_y_pred_with_opt_hyperplane = train_val_y_pred_with_optimal_hyperplane[:training_set]
                    valid_y_pred_with_opt_hyperplane = train_val_y_pred_with_optimal_hyperplane[training_set:]
                    step_train_acc_opt, step_train_pre_opt, step_train_f1_opt, step_train_rec_opt, step_train_auc_opt = self.compute_score(
                        y_true=y_train[:training_set], y_pred=train_y_pred_with_opt_hyperplane)

                    train_y_pred_1 = train_val_y_pred_1[:training_set]
                    valid_y_pred_1 = train_val_y_pred_1[training_set:]
                    step_train_acc_1, step_train_pre_1, step_train_f1_1, step_train_rec_1, step_train_auc_1 = self.compute_score(
                        y_true=y_train[:training_set], y_pred=train_y_pred_1)

                    message = "[train] total_loss %.5f\n" % step_loss
                    message += "[train] cost_sensitive_loss %.5f\n" % train_val_min_cost_value

                    message += "[train] accuracy_0 %.2f\n" % (step_train_acc_0 * 100)
                    message += "[train] precision_0 %.2f\n" % (step_train_pre_0 * 100)
                    message += "[train] f1_0 %.2f\n" % (step_train_f1_0 * 100)
                    message += "[train] recall_0 %.2f\n" % (step_train_rec_0 * 100)
                    message += "[train] auc_0 %.2f\n" % (step_train_auc_0 * 100)

                    message += "[train] accuracy_opt %.2f\n" % (step_train_acc_opt * 100)
                    message += "[train] precision_opt %.2f\n" % (step_train_pre_opt * 100)
                    message += "[train] f1_opt %.2f\n" % (step_train_f1_opt * 100)
                    message += "[train] recall_opt %.2f\n" % (step_train_rec_opt * 100)
                    message += "[train] auc_opt %.2f\n" % (step_train_auc_opt * 100)

                    message += "[train] accuracy_1 %.2f\n" % (step_train_acc_1 * 100)
                    message += "[train] precision_1 %.2f\n" % (step_train_pre_1 * 100)
                    message += "[train] f1_1 %.2f\n" % (step_train_f1_1 * 100)
                    message += "[train] recall_1 %.2f\n" % (step_train_rec_1 * 100)
                    message += "[train] auc_1 %.2f\n" % (step_train_auc_1 * 100)

                    message += "[train] n_data_in_strip %d\n" % train_val_n_data_in_strip
                    utils.print_and_write_logging_file(self.logging_path, message, self.running_mode, show_message=False)

                    step_val_acc_0, step_val_pre_0, step_val_f1_0, step_val_rec_0, step_val_auc_0 = self.compute_score(
                        y_true=y_valid[:dev_set], y_pred=valid_y_pred_0)
                    step_val_acc_opt, step_val_pre_opt, step_val_f1_opt, step_val_rec_opt, step_val_auc_opt = self.compute_score(
                        y_true=y_valid[:dev_set], y_pred=valid_y_pred_with_opt_hyperplane)
                    step_val_acc_1, step_val_pre_1, step_val_f1_1, step_val_rec_1, step_val_auc_1 = self.compute_score(
                        y_true=y_valid[:dev_set], y_pred=valid_y_pred_1)

                    message = "[eval] total_loss %.5f\n" % average_dev_loss
                    message += "[eval] cost_sensitive_loss %.5f\n" % train_val_min_cost_value

                    message += "[eval] accuracy_0 %.2f\n" % (step_val_acc_0 * 100)
                    message += "[eval] precision_0 %.2f\n" % (step_val_pre_0 * 100)
                    message += "[eval] f1_0 %.2f\n" % (step_val_f1_0 * 100)
                    message += "[eval] recall_0 %.2f\n" % (step_val_rec_0 * 100)
                    message += "[eval] auc_0 %.2f\n" % (step_val_auc_0 * 100)

                    message += "[eval] accuracy_opt %.2f\n" % (step_val_acc_opt * 100)
                    message += "[eval] precision_opt %.2f\n" % (step_val_pre_opt * 100)
                    message += "[eval] f1_opt %.2f\n" % (step_val_f1_opt * 100)
                    message += "[eval] recall_opt %.2f\n" % (step_val_rec_opt * 100)
                    message += "[eval] auc_opt %.2f\n" % (step_val_auc_opt * 100)

                    message += "[eval] accuracy_1 %.2f\n" % (step_val_acc_1 * 100)
                    message += "[eval] precision_1 %.2f\n" % (step_val_pre_1 * 100)
                    message += "[eval] f1_1 %.2f\n" % (step_val_f1_1 * 100)
                    message += "[eval] recall_1 %.2f\n" % (step_val_rec_1 * 100)
                    message += "[eval] auc_1 %.2f\n" % (step_val_auc_1 * 100)

                    message += "[eval] n_data_in_strip %d\n" % train_val_n_data_in_strip
                    message += "-----------------------------------------------------\n"
                    utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)

                    step_time, step_loss = 0.0, 0.0  # it is important to set step_time and step_loss return to zero.

            writer.close()
        message = "Finish training process.\n"
        utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)

    def test(self, checkpoint_path, x_test_opcode, x_test_assembly, x_test_seq_len, y_test):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(config=utils.get_default_config()) as sess:
                check_point = tf.train.get_checkpoint_state(checkpoint_path)

                try:
                    saver = tf.train.import_meta_graph("{}.meta".format(check_point.model_checkpoint_path))
                    saver.restore(sess, check_point.model_checkpoint_path)
                except:
                    print("Can not find the saved model.")

                message = "Loaded model parameters from %s\n" % check_point.model_checkpoint_path
                utils.print_and_write_logging_file(self.logging_path, message, self.running_mode, self.datetime)

                # get the placeholders from the graph by name
                X_opcode = graph.get_operation_by_name("input/x_opcode_input").outputs[0]
                X_assembly = graph.get_operation_by_name("input/x_assemply_input").outputs[0]
                Y = graph.get_operation_by_name("input/true_label").outputs[0]
                sequence_length = graph.get_operation_by_name("input/seq_length").outputs[0]

                # get tensor to visualize
                phi_x_tilde = graph.get_operation_by_name("oc-svm/phi_x_tilde").outputs[0]
                # get tensor for prediction
                w_phi_minus_rho2 = graph.get_operation_by_name("oc-svm/sub").outputs[0]

                test_set = x_test_opcode.shape[0] - x_test_opcode.shape[0] % self.batch_size
                test_batches = utils.make_batches(test_set, self.batch_size)

                full_phi_x_tilde = np.zeros((test_set, 2*self.num_random_features))  # (batch_size, 2*n_random_features)
                test_full_pred = np.array([])  # (batch_size, 1)
                for batch_idx, (batch_start, batch_end) in enumerate(test_batches):
                    test_x_opcode = utils.convert_list_sparse_to_dense(x_test_opcode[batch_start:batch_end])
                    test_x_assembly = utils.convert_list_sparse_to_dense(x_test_assembly[batch_start:batch_end])
                    test_y = y_test[batch_start:batch_end]
                    test_seq_len = x_test_seq_len[batch_start:batch_end]

                    test_feed_dict = {
                        X_opcode: test_x_opcode,
                        X_assembly: test_x_assembly,
                        Y: test_y,
                        sequence_length: test_seq_len,
                    }

                    batch_phi_x_tilde, test_batch_pred = sess.run([phi_x_tilde, w_phi_minus_rho2],
                                                                feed_dict=test_feed_dict)
                    test_full_pred = np.append(test_full_pred, test_batch_pred)
                    full_phi_x_tilde[batch_start:batch_end] = batch_phi_x_tilde

                test_min_cost_value, test_y_pred_0, test_y_pred_with_optimal_hyperplane, test_y_pred_1, test_n_data_in_strip = self.compute_cost_sensitive_loss_and_y_pred(
                    test_full_pred, y_test[:test_set])

                test_acc_0, test_pre_0, test_f1_0, test_rec_0, test_auc_0 = self.compute_score(
                    y_true=y_test[:test_set],
                    y_pred=test_y_pred_0)

                test_acc_opt, test_pre_opt, test_f1_opt, test_rec_opt, test_auc_opt = self.compute_score(
                    y_true=y_test[:test_set],
                    y_pred=test_y_pred_with_optimal_hyperplane)

                test_acc_1, test_pre_1, test_f1_1, test_rec_1, test_auc_1 = self.compute_score(
                    y_true=y_test[:test_set],
                    y_pred=test_y_pred_1)

                message = "[test] cost_sensitive_loss %.5f\n" % test_min_cost_value

                message += "[test] accuracy_0 %.2f\n" % (test_acc_0 * 100)
                message += "[test] precision_0 %.2f\n" % (test_pre_0 * 100)
                message += "[test] f1_0 %.2f\n" % (test_f1_0 * 100)
                message += "[test] recall_0 %.2f\n" % (test_rec_0 * 100)
                message += "[test] auc_0 %.2f\n" % (test_auc_0 * 100)

                message += "[test] accuracy_opt %.2f\n" % (test_acc_opt * 100)
                message += "[test] precision_opt %.2f\n" % (test_pre_opt * 100)
                message += "[test] f1_opt %.2f\n" % (test_f1_opt * 100)
                message += "[test] recall_opt %.2f\n" % (test_rec_opt * 100)
                message += "[test] auc_opt %.2f\n" % (test_auc_opt * 100)

                message += "[test] accuracy_1 %.2f\n" % (test_acc_1 * 100)
                message += "[test] precision_1 %.2f\n" % (test_pre_1 * 100)
                message += "[test] f1_1 %.2f\n" % (test_f1_1 * 100)
                message += "[test] recall_1 %.2f\n" % (test_rec_1 * 100)
                message += "[test] auc_1 %.2f\n" % (test_auc_1 * 100)

                message += "[test] n_data_in_strip %d\n" % test_n_data_in_strip
                message += "-----------------------------------------------------\n"
                utils.print_and_write_logging_file(self.logging_path, message, self.running_mode)


def main():
    model = DCKM()
    if model.running_mode == 1:  # train
        if model.data_size == 'full_dataset':
            print("Choosing full dataset ... \n")
            list_arch_os = ['32-windows', '64-windows', '32-ubuntu', '64-ubuntu']
        elif model.data_size == 'windows_only':
            print("Choosing data from windows ... \n")
            list_arch_os = ['32-windows', '64-windows']
        elif model.data_size == 'ubuntu_only':
            print("Choosing data from ubuntu ... \n")
            list_arch_os = ['32-ubuntu', '64-ubuntu']

        x_train_opcode, x_train_assembly, x_train_seq_len, y_train, \
        x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid, \
        x_test_opcode, x_test_assembly, x_test_seq_len, y_test, model.time_steps, model.vocab_opcode_size = utils.load_dataset(
            list_arch_os, model.dataset_name)

        model.build_model()
        model.train(x_train_opcode, x_train_assembly, x_train_seq_len, y_train,
                    x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid)
    elif model.running_mode == 0:  # test
        if model.data_size == 'full_dataset':
            list_arch_os = ['32-windows', '64-windows', '32-ubuntu', '64-ubuntu']
        else:
            list_arch_os = ['32-windows']

        _, _, _, _, \
        _, _, _, _, \
        x_test_opcode, x_test_assembly, x_test_seq_len, y_test, model.time_steps, model.vocab_opcode_size = \
            utils.load_dataset(list_arch_os, model.dataset_name)

        model.build_model()

        checkpoint_path = 'saved-model/full_dataset/2018-29-10-15-14-40'  # set your saved-model's path
        model.test(checkpoint_path, x_test_opcode, x_test_assembly, x_test_seq_len, y_test)


if __name__ == '__main__':
    main()