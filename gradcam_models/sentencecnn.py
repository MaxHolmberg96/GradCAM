from gradcam_models.gradcam import gradcam
import tensorflow as tf
import cv2
import numpy as np
import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import matplotlib.pyplot as plt


def clean_str(in_str):
    in_str = str(in_str)
    # replace urls with 'url'
    in_str = re.sub(
        r"(https?://(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+["
        r"a-zA-Z0-9]\.[^\s]{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})",
        "url",
        in_str,
    )
    in_str = re.sub(r"([^\s\w]|_)+", "", in_str)
    return in_str.strip().lower()


class sentencecnn(gradcam):
    def __init__(self, train_path, word2vec_embeddings_path):
        np.random.seed(0)
        super().__init__()
        self.df = pd.read_csv(train_path, delimiter="\t")
        self.df["text"] = self.df["Phrase"].apply(clean_str)
        self.df_0 = self.df[self.df["Sentiment"] == 0].sample(frac=1)
        self.df_1 = self.df[self.df["Sentiment"] == 1].sample(frac=1)
        self.df_2 = self.df[self.df["Sentiment"] == 2].sample(frac=1)
        self.df_3 = self.df[self.df["Sentiment"] == 3].sample(frac=1)
        self.df = self.df[self.df["Sentiment"] == 4].sample(frac=1)
        # we want a balanced set for training against - there are 7072 `0` examples
        self.sample_size = 7072
        self.sequence_length = 52
        self.data = pd.concat(
            [
                self.df_0.head(self.sample_size),
                self.df_1.head(self.sample_size),
                self.df_2.head(self.sample_size),
                self.df_3.head(self.sample_size),
                self.df.head(self.sample_size),
            ]
        ).sample(frac=1)
        self.data["l"] = self.data["Phrase"].apply(lambda x: len(str(x).split(" ")))

        # Create tokenizer
        self.max_features = 20000  # this is the number of words we care about
        self.tokenizer = Tokenizer(num_words=self.max_features, split=" ", oov_token="<unw>")
        self.tokenizer.fit_on_texts(self.data["Phrase"].values)

        self.embedding_dim = 200  # Kim Yoon uses 300 here
        self.num_filters = 100

        self.embeddings_index = {}
        f = open(word2vec_embeddings_path, encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            self.embeddings_index[word] = coefs
        f.close()

        print("Loaded %s word vectors." % len(self.embeddings_index))
        self.word_index = self.tokenizer.word_index

        # Create embedding index to use in Embedding layer in model
        self.num_words = min(self.max_features, len(self.word_index)) + 1

        # first create a matrix of zeros, this is our embedding matrix
        self.embedding_matrix = np.zeros((self.num_words, self.embedding_dim))

        # for each word in our tokenizer lets try to find that work in our w2v model
        for word, i in self.word_index.items():
            if i > self.max_features:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # we found the word - add that words vector to the matrix
                self.embedding_matrix[i] = embedding_vector
            else:
                # doesn't exist, assign a random vector
                self.embedding_matrix[i] = np.random.randn(self.embedding_dim)

    def load_data(self):
        self.sst2_data = pd.concat(
            [
                self.df_0.head(self.sample_size),
                self.df_1.head(self.sample_size),
                self.df_3.head(self.sample_size),
                self.df.head(self.sample_size),
            ]
        ).sample(frac=1)

        def merge_sentiments(x):
            if x == 0 or x == 1:
                return 0
            else:
                return 1

        self.sst2_data["Sentiment"] = self.sst2_data["Sentiment"].apply(merge_sentiments)

        self.sst2_tokenizer = Tokenizer(num_words=self.max_features, split=" ", oov_token="<unw>")
        self.sst2_tokenizer.fit_on_texts(self.sst2_data["Phrase"].values)
        self.sst2_word_index = self.sst2_tokenizer.word_index

        self.sst2_X = self.sst2_tokenizer.texts_to_sequences(self.sst2_data["Phrase"].values)
        self.sst2_X = pad_sequences(self.sst2_X, self.sequence_length)

        self.sst2_y = tf.one_hot(self.sst2_data["Sentiment"].values, 2)

        self.sst2_X_train = self.sst2_X
        self.sst2_y_train = self.sst2_y

    def initalize_model(self, softmax=True):
        inputs = Input(shape=(self.sequence_length,), dtype="int32")
        embedding_layer = Embedding(
            self.num_words,
            self.embedding_dim,
            embeddings_initializer=Constant(self.embedding_matrix),
            input_length=self.sequence_length,
            trainable=True,
        )(inputs)

        reshape = Reshape((self.sequence_length, self.embedding_dim, 1))(embedding_layer)

        conv_0 = Conv2D(
            self.num_filters,
            kernel_size=(3, self.embedding_dim),
            padding="valid",
            kernel_initializer="normal",
            activation="relu",
            kernel_regularizer=L2(3),
        )(reshape)
        conv_1 = Conv2D(
            self.num_filters,
            kernel_size=(4, self.embedding_dim),
            padding="valid",
            kernel_initializer="normal",
            activation="relu",
            kernel_regularizer=L2(3),
        )(reshape)
        conv_2 = Conv2D(
            self.num_filters,
            kernel_size=(5, self.embedding_dim),
            padding="valid",
            kernel_initializer="normal",
            activation="relu",
            kernel_regularizer=L2(3),
        )(reshape)

        maxpool_0 = MaxPool2D(pool_size=(self.sequence_length - 3 + 1, 1), strides=(1, 1), padding="valid")(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(self.sequence_length - 4 + 1, 1), strides=(1, 1), padding="valid")(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(self.sequence_length - 5 + 1, 1), strides=(1, 1), padding="valid")(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)

        dropout = Dropout(0.5)(flatten)
        # note the different activation
        if softmax:
            output = Dense(units=2, activation="softmax")(dropout)
        else:
            output = Dense(units=2, activation=None)(dropout)

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.summary()

    def load_weights(self, ckpt_path):
        self.initalize_model(softmax=False)
        self.model.load_weights(ckpt_path)
        print("Loaded weights for model without softmax")

    def train(self, batch_size=32, validation_split=0.2, epochs=30, verbose=1):

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="sentiment_cnn_weights/cp.ckpt", save_weights_only=True, verbose=1
        )
        print("Training model with softmax")
        self.history = self.model.fit(
            self.sst2_X_train,
            self.sst2_y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
            callbacks=[cp_callback],
        )

        self.initalize_model(softmax=False)
        self.model.load_weights("sentiment_cnn_weights/cp.ckpt")
        print("Loaded weights for model without softmax")

    def plot_training_history(self):
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()

        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()

    def convert_sentence(self, sentence):
        x = self.sst2_tokenizer.texts_to_sequences([sentence])
        return pad_sequences(x, self.sequence_length)[0]

    def predict(self, X):
        return tf.nn.softmax(self.model.predict(X)).numpy()

    def summary(self):
        self.model.summary()

    def get_heatmap(self, c, data, conv_layer):
        # Get final convolutional layer
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(conv_layer).output, self.model.output],
        )

        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(data)
            y_c = predictions[:, c]

        conv_output = conv_outputs[0]
        grads = tape.gradient(y_c, conv_outputs)
        grads = grads[0]

        # Calculate gradcam heatmap
        alphas = tf.reduce_mean(grads, axis=[0, 1])
        linear_combinaton = tf.reduce_sum(alphas * conv_output, axis=-1)
        linear_combinaton = cv2.resize(
            linear_combinaton.numpy(), (data.shape[0], data.shape[1]), interpolation=cv2.INTER_CUBIC
        )
        cam = tf.nn.relu(linear_combinaton).numpy()
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam
