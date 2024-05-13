import os
import time
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

import toolkit

class BertModel(object):
    def __init__(self) -> None:
        self.tokeniser, self.model = None, None
        self.load_model(toolkit.get_dir() + '/models/')

    # Save the tokeniser and model
    def save_model(self, path: str, force: bool = False) -> bool:
        if not force and (os.path.exists(path + '/tokeniser') or os.path.exists(path + '/model')):
            return False
        try:
            self.tokeniser.save_pretrained(path + '/tokeniser') # Save tokeniser
            self.model.save_pretrained(path + '/model') # Save model
            toolkit.console(f"Model saved at {path}")
        except Exception as e:
            toolkit.error(f"Could not save model. {e}")

    # Load the tokeniser and model
    def load_model(self, path: str) -> None:
        try:
            self.tokeniser = BertTokenizer.from_pretrained(path + '/tokeniser') # Load tokeniser
            self.model = TFBertForSequenceClassification.from_pretrained(path + '/model') # Load model
            toolkit.console(f"Model loaded from {path}")
        except Exception as e:
            toolkit.error(f"Could not load model. {e}")
            self.new_model()

    def new_model(self):
        """
        Initialise a new tokeniser and model.
        """
        # Initialize a new tokeniser
        self.tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Initialize a new model
        self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def tokenise(self, text: list[str]) -> dict:
        """
        Tokenise and encode the data using BERT tokeniser.
        @param text: List of strings to be tokenised.
        @return: Dictionary containing tokenised inputs.
        """
        if self.tokeniser is None:
            self.new_model()

        return self.tokeniser.batch_encode_plus(text, padding=True, truncation=True, max_length=128, return_tensors='tf')

    def fit_model(self, X_train_encoded, y_train, X_val_encoded, y_val) -> None:
        """
        Fit and evaluate the BERT model.
        @param X_train_encoded: Dictionary containing tokenised inputs for training data.
        @param y_train: List of labels for training data.
        @param X_val_encoded: Dictionary containing tokenised inputs for validation data.
        @param y_val: List of labels for validation data.
        """
        # Compile the model with optimiser, loss function, and metrics
        optimiser = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=optimiser, loss=loss, metrics=[metric])

        # Train the model
        history = self.model.fit([X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']], np.array(y_train),
                                validation_data=([X_val_encoded['input_ids'], X_val_encoded['token_type_ids'], X_val_encoded['attention_mask']], np.array(y_val)),
                                batch_size=32, epochs=3)
        
        return history

    def train(self, dataset: pd.DataFrame, path: str) -> None:
        """
        Train the BERT model.
        @param dataset: DataFrame containing 'text' and 'sentiment' columns.
        """
        t = time.time() # Start timer
        toolkit.console("Started training...")

        text = dataset['text'].tolist()
        sentiment = dataset['sentiment'].tolist()

        # Split the data into 80% training, 10% testing, and 10% validation
        X_train, X_test, y_train, y_test = train_test_split(text, sentiment, test_size = 0.2, stratify=sentiment, random_state = 0)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=0)

        self.new_model()

        # Tokenise and encode the data
        X_train_encoded = self.tokenise(X_train)
        X_val_encoded = self.tokenise(X_val)
        X_test_encoded = self.tokenise(X_test)

        # Check the encoded dataset
        k = random.randint(0, len(X_train) - 1)
        print()
        toolkit.console(f"Information on random row - {k}.")
        toolkit.console(f"Training Comments - {X_train[k]}\n")
        toolkit.console(f"Input Ids - \n{X_train_encoded['input_ids'][k]}\n")
        toolkit.console(f"Decoded Ids - \n{self.tokeniser.decode(X_train_encoded['input_ids'][k])}\n")
        toolkit.console(f"Attention Mask - \n{X_train_encoded['attention_mask'][k]}\n")
        toolkit.console(f"Labels - {y_train[k]}\n")

        # Fit the model or perform cross-validation
        if toolkit.get_config('cross_validation'):
            self.cross_validate(dataset)
        else:
            history = self.fit_model(X_train_encoded, y_train, X_val_encoded, y_val)
            self.plot_training_history(history)
        
        toolkit.console(f"Finished training in {round(time.time()-t)} seconds.\n")

        # Evaluate the model on the test set
        test_loss, test_accuracy, pred_labels, actual_labels = self.test(X_test_encoded, y_test)

        # Plot confusion matrix
        self.plot_confusion_matrix(actual_labels, pred_labels)

        # Save the trained model
        self.save_model(path, force=True)

    def cross_validate(self, dataset: pd.DataFrame, n_splits: int = 5) -> None:
        """
        Perform cross-validation on the dataset.
        @param dataset: DataFrame containing 'text' and 'sentiment' columns.
        @param n_splits: Number of splits for cross-validation.
        """
        text = dataset['text'].tolist()
        sentiment = dataset['sentiment'].tolist()

        # Define cross-validation splitter
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Initialize lists to store results
        test_losses = []
        test_accuracies = []
        train_accuracies = []
        pred_labels_list = []
        actual_labels_list = []

        # Perform cross-validation
        k = 0
        for train_index, val_index in kf.split(text):
            t = time.time() # Start timer
            toolkit.console(f"Started fold {k}...")

            X_train, X_val = [text[i] for i in train_index], [text[i] for i in val_index]
            y_train, y_val = [sentiment[i] for i in train_index], [sentiment[i] for i in val_index]

            # Tokenise and encode the data
            X_train_encoded = self.tokenise(X_train)
            X_val_encoded = self.tokenise(X_val)

            # Fit and evaluate the model
            history = self.fit_model(X_train_encoded, y_train, X_val_encoded, y_val)

            # Evaluate the model
            test_loss, test_accuracy, pred_labels, actual_labels = self.test(X_val_encoded, y_val)

            # Append results to lists
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            train_accuracies.append(history.history['accuracy'][-1])  # Get the last accuracy from training history
            pred_labels_list.extend(pred_labels)
            actual_labels_list.extend(actual_labels)

            toolkit.console(f"Finished fold {k} in {round(time.time()-t)} seconds.\n")
            k += 1

        # Calculate average metrics
        avg_test_loss = np.mean(test_losses)
        avg_test_accuracy = np.mean(test_accuracies)

        # Print and log the results
        toolkit.console(f"Average Test Loss: {avg_test_loss}")
        toolkit.console(f"Average Test Accuracy: {avg_test_accuracy}")

        # Print classification report
        toolkit.console("Classification Report:")
        print(classification_report(actual_labels_list, pred_labels_list))

        # Plot train accuracy vs test accuracy
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Train Accuracy vs Test Accuracy')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def test(self, X_test_encoded: dict, y_test: list[int]) -> tuple[float, float, list[str], list[str]]:
        """
        Evaluate the BERT model.
        @param X_test_encoded: Dictionary containing tokenised inputs for test data.
        @param y_test: List of labels for test data.
        @return: Test loss, test accuracy, predicted labels, and actual labels.
        """
        if self.model is None:
            raise ValueError("Model not initialised. Load or initialise the model first.")

        t = time.time() # Start timer
        toolkit.console("Started testing...")

        # Evaluate the model on test data
        test_loss, test_accuracy = self.model.evaluate([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']], np.array(y_test))

        pred = self.model.predict([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])
        logits = pred.logits
        pred_labels = tf.argmax(logits, axis=1)  # Use argmax along the appropriate axis to get predicted labels
        pred_labels = pred_labels.numpy() # Convert the predicted labels to a numpy array

        labels = {1: 'Positive', 0: 'Negative'}

        # Map the predicted labels to the actual labels
        pred_labels = [labels[i] for i in pred_labels]
        actual_labels = [labels[i] for i in y_test]

        # Print and log the results
        toolkit.console(f"Test Loss: {test_loss * 100}%")
        toolkit.console(f"Test Accuracy: {test_accuracy * 100}%")
        print(classification_report(actual_labels, pred_labels))
        toolkit.console(f"Finished testing in {round(time.time()-t)} seconds.\n")

        return test_loss, test_accuracy, pred_labels, actual_labels

    def predict(self, text: list[str]) -> str:
        """
        Predict the sentiment of text using the trained BERT model.
        @param text: List of strings to predict sentiment for.
        @return: Predicted sentiment label.
        """
        if self.model is None or self.tokeniser is None:
            raise ValueError("Model or tokeniser not initialised. Load or initialise them first.")
        
        t = time.time() # Start timer
        toolkit.console("Started predicting...")
        
        confidence_threshold = toolkit.get_config('confidence_threshold')

        input_ids, token_type_ids, attention_mask = self.tokenise(text).values()
        prediction = self.model.predict([np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)])

        labels = {1: 'Positive', 0: 'Negative'}

        pred_logits = prediction.logits
        pred_probs = tf.nn.softmax(pred_logits, axis=1).numpy()
        pred_confidence = np.max(pred_probs, axis=1)

        pred_label_indices = np.argmax(pred_logits, axis=1)
        pred_labels = [labels[i] if confidence >= confidence_threshold else 'Neutral' for i, confidence in zip(pred_label_indices, pred_confidence)]

        truncated_texts = [txt[:64] + '...' if len(txt) > 64 else txt for txt in text] # Truncate texts for logging
        # Log the prediction and confidence in the console
        for txt, label, confidence in zip(truncated_texts, pred_labels, pred_confidence):
            confidence_percent = confidence * 100
            toolkit.console(f"{txt:<67} predicted {label.upper()} with a confidence of {confidence_percent:.2f}%")

        toolkit.console(f"Finished predicting in {round(time.time()-t)} seconds.\n")

        return pred_labels # Return the labels
    
    def plot_training_history(self, history):
        """
        Plot training and validation accuracy and loss.
        @param history: Training history object returned by model.fit().
        """
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix.
        @param y_true: True labels.
        @param y_pred: Predicted labels.
        """
        labels = ['Negative', 'Positive']
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
