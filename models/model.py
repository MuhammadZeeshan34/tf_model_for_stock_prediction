import tensorflow as tf


class TF_Model:

    def __init__(self, x_training, y_training, x_test, y_test, learning_rate = 0.01, no_epochs = 100, no_inputs = 1):
        self.learning_rate = learning_rate
        self.no_epochs - no_epochs
        self.number_of_inputs = no_epochs
        self.layer1_nodes = 50
        self.layer2_nodes = 100
        self.layer3_nodes = 50
        self.cost = None
        self.optimizer = None
        self.x_train = x_training
        self.y_train = y_training
        self.x_test = x_test
        self.y_test = y_test


    def graph_creation(self):
        # design input layer
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=(None, self.number_of_inputs))

        # design first layer
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("weights1",shape=[self.number_of_inputs,self.layer1_nodes])
            biases = tf.get_variable("biases1",shape=[self.number_of_inputs,self.layer1_nodes],initializer=tf.zeros_initializer())
            output_of_layer1 = tf.nn.relu(tf.matmul(X,weights)+biases)

        # design of layer 2
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("weights2",shape=[self.layer1_nodes,self.layer2_nodes])
            biases = tf.get_variable("biases2",shape=[self.layer1_nodes,self.layer2_nodes],initializer=tf.zeros_initializer())
            output_of_layer2 = tf.nn.relu(tf.matmul(output_of_layer1,weights)+biases)

        with tf.variable_scope('layer3'):
            weights = tf.get_variable("weights3",shape=[self.layer2_nodes,self.layer3_nodes])
            biases = tf.get_variable("biases3",shape=[self.layer2_nodes,self.layer3_nodes],initializer=tf.zeros_initializer())
            output_of_layer3 = tf.nn.relu(tf.matmul(output_of_layer2,weights)+biases)

        with tf.variable_scope('output'):
            weights = tf.get_variable("weights3",shape=[self.layer3_nodes,1])
            biases = tf.get_variable("biases3",shape=[self.layer3_nodes,1],initializer=tf.zeros_initializer())
            prediction = tf.matmul(output_of_layer3,weights)+biases



        # cost function design
        with tf.variable_scope('cost'):
            self.Y = tf.placeholder(tf.float32,shape=(None,1))
            self.cost = tf.reduce_mean(tf.squared_difference(prediction,self.Y))

        # optimizer define
        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


    def train(self):
        if self.cost == None or self.optimizer == None:
            print("Construct graph before training model")
            return

        with tf.Session() as session:

            # Global variables initializer
            session.run(tf.global_variables_initializer())

            for epoch in range(self.no_epochs):
                session.run(self.optimizer, feed_dict={self.X: self.x_train, self.Y: self.y_train})

                # Essential logging
                if epoch % 5 == 0:
                    training_cost = session.run(self.cost, feed_dict={self.X : self.x_train, self.Y : self.y_train})
                    testing_cost = session.run(self.cost, feed_dict={self.X : self.x_test, self.Y : self.y_test})

                    print("Epoch:" + str(epoch)," Training cost:" + str(training_cost),
                                        " Testing cost:"+str(testing_cost))

            print("Training completed")

            final_training_cost = session.run(self.cost, feed_dict={self.X: self.x_train, self.Y: self.y_train})
            final_testing_cost = session.run(self.cost, feed_dict={self.X: self.x_test, self.Y: self.y_test})

            print("Final training cost. {}".format(final_testing_cost))
            print("Final testing cost. {}".format(final_testing_cost))






