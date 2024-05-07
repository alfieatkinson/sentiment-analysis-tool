import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

import toolkit

class BertModel(object):
    def __init__(self) -> None:
        self.tokeniser, self.model = None, None
        self.load_model(toolkit.get_dir() + '/models')

    # Save the tokeniser and model
    def save_model(self, path: str, force: bool = False) -> bool:
        if not force and (os.path.exists(path + '/tokeniser') or os.path.exists(path + '/model')):
            return False
        try:
            self.tokeniser.save_pretrained(path + '/tokeniser') # Save tokeniser
            self.model.save_pretrained(path + '/model') # Save model
        except:
            toolkit.alert("Could not save model.")

    # Load the tokeniser and model
    def load_model(self, path: str) -> None:
        try:
            self.tokeniser = BertTokenizer.from_pretrained(path + '/tokeniser') # Load tokeniser
            self.model = TFBertForSequenceClassification.from_pretrained(path + '/model') # Load model
        except:
            toolkit.alert("Could not load model.")
            self.tokeniser, self.model = None, None

    # Split and train the model
    def train(self) -> None:
        t = time.time() # Start timer
        print(f"Start time: {toolkit.time()}")

        text, sentiment = toolkit.initialise_dataset()

        # Split the data into 95% training and 5% testing
        X_train, X_test, y_train, y_test = train_test_split(text, sentiment, test_size = 0.05, stratify=sentiment, random_state = 0)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=0)
        print()
        print("Data split done.")

        # Tokenise and encode the data using BERT tokeniser
        self.tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        max_len= 128
        # Tokenise and encode the sentences
        X_train_encoded = self.tokeniser.batch_encode_plus(X_train, padding=True, truncation=True, max_length = max_len, return_tensors='tf')
        X_val_encoded = self.tokeniser.batch_encode_plus(X_val, padding=True, truncation=True, max_length = max_len, return_tensors='tf')
        X_test_encoded = self.tokeniser.batch_encode_plus(X_test, padding=True, truncation=True, max_length = max_len, return_tensors='tf')

        # Check the encoded dataset
        k = 0
        print()
        print('Training Comments: ', X_train[k])
        print('\nInput Ids: \n', X_train_encoded['input_ids'][k])
        print('\nDecoded Ids: \n', self.tokeniser.decode(X_train_encoded['input_ids'][k]))
        print('\nAttention Mask: \n', X_train_encoded['attention_mask'][k])
        print('\nLabels: ', y_train[k])

        # Initialise the model
        self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # Compile the model with optimiser, loss function, and metrics
        optimiser = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=optimiser, loss=loss, metrics=[metric])

        # Train the model
        history = self.model.fit([np.array(X_train_encoded['input_ids']), np.array(X_train_encoded['token_type_ids']), np.array(X_train_encoded['attention_mask'])], np.array(y_train), validation_data=([np.array(X_val_encoded['input_ids']), np.array(X_val_encoded['token_type_ids']), np.array(X_val_encoded['attention_mask'])], np.array(y_val)), batch_size=32, epochs=3)

        print()
        print(f'Training complete.')
        print(f"Finish time: {toolkit.time()}") # End timer
        print(f'Time Taken: {round(time.time()-t)} seconds')

        # Evaluate the model on test data
        test_loss, test_accuracy = self.model.evaluate([np.array(X_test_encoded['input_ids']), np.array(X_test_encoded['token_type_ids']), np.array(X_test_encoded['attention_mask'])], np.array(y_test))

        print()
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_accuracy}")
        print("Evaluation complete.")

        pred = self.model.predict([np.array(X_test_encoded['input_ids']), np.array(X_test_encoded['token_type_ids']), np.array(X_test_encoded['attention_mask'])])
        logits = pred.logits

        pred_labels = tf.argmax(logits, axis=1) # Use argmax along the appropriate axis to get predicted labels
        pred_labels = pred_labels.numpy() # Convert the predicted labels to a numpy array

        labels = {
            1: 'Positive',
            0: 'Negative'
        }

        # Map the predicted labels to the actual labels
        pred_labels = [labels[i] for i in pred_labels]
        actual_labels = [labels[i] for i in y_test]

        print()
        print(f"Predicted labels: {pred_labels[:10]}")
        print(f"Actual labels: {actual_labels[:10]}")
        print()
        print("Classification report: \n", classification_report(actual_labels, pred_labels))
        print("Testing complete.")

    def predict(self, text: str) -> str:
        t = time.time() # Start timer
        print()
        print(f"Start time: {toolkit.time()}")

        toolkit.clean(text) # Clean the text

        input_ids, token_type_ids, attention_mask = self.tokeniser.batch_encode_plus(text, padding=True, truncation=True, max_length=128, return_tensors='tf').values()
        prediction = self.model.predict([np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)])

        labels = {
            1: 'Positive',
            0: 'Negative'
        }

        pred_label = tf.argmax(prediction.logits, axis=1) # Use argmax along the appropriate axis to get the predicted labels
        prediction = labels[pred_label.numpy()[0]] # Choose a sentiment label based on the TensorFlow tensor

        #print(f"Predicted: {prediction}")

        print(f"Finish time: {toolkit.time()}") # End timer
        print(f'Time Taken: {round(time.time()-t)} seconds')

        return prediction