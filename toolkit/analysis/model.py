import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

import toolkit

class BertModel(object):
    def __init__(self) -> None:
        self.tokeniser, self.model = None, None
        self.load_model(f'{toolkit.get_dir()}/models')

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

    def tokenise(self, text: list[str]) -> dict:
        """
        Tokenise and encode the data using BERT tokeniser.
        @param text: List of strings to be tokenised.
        @return: Dictionary containing tokenised inputs.
        """
        if self.tokeniser is None:
            raise ValueError("Tokeniser not initialised. Load or initialise the tokeniser first")

        max_len = 128
        return self.tokeniser.batch_encode_plus(text, padding=True, truncation=True, max_length=max_len, return_tensors='tf')

    def train(self, dataset: pd.DataFrame) -> None:
        """
        Train the BERT model.
        @param dataset: DataFrame containing 'text' and 'sentiment' columns.
        """
        t = time.time() # Start timer
        toolkit.console("\nStarted training...")

        text = dataset['Text'].tolist()
        sentiment = dataset['Sentiment'].tolist()

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(text, sentiment, test_size=0.1, random_state=42)

        # Initialize a new tokeniser
        self.tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')

        # Initialize a new model
        self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

        # Tokenise and encode the data
        X_train_encoded = self.tokenise(X_train)
        X_val_encoded = self.tokenise(X_val)

        # Compile the model with optimiser, loss function, and metrics
        optimiser = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=optimiser, loss=loss, metrics=[metric])

        # Train the model
        history = self.model.fit([X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']], np.array(y_train),
                                validation_data=([X_val_encoded['input_ids'], X_val_encoded['token_type_ids'], X_val_encoded['attention_mask']], np.array(y_val)),
                                batch_size=32, epochs=3)
        
        toolkit.console(f"Finished training in {round(time.time()-t)} seconds.")

        # Save the trained model
        self.save_model(f'{toolkit.get_dir()}/models')

        # Evaluate the model on the validation set
        self.test(X_val_encoded, y_val)

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
        pred_labels_list = []
        actual_labels_list = []

        # Perform cross-validation
        for train_index, test_index in kf.split(text):
            X_train, X_test = [text[i] for i in train_index], [text[i] for i in test_index]
            y_train, y_test = [sentiment[i] for i in train_index], [sentiment[i] for i in test_index]

            # Tokenise and encode the data
            X_train_encoded = self.tokenise(X_train)
            X_test_encoded = self.tokenise(X_test)

            # Train the model
            self.train(X_train_encoded, y_train, X_test_encoded, y_test)

            # Evaluate the model
            test_loss, test_accuracy, pred_labels, actual_labels = self.test(X_test_encoded, y_test)

            # Append results to lists
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            pred_labels_list.extend(pred_labels)
            actual_labels_list.extend(actual_labels)

        # Calculate average metrics
        avg_test_loss = np.mean(test_losses)
        avg_test_accuracy = np.mean(test_accuracies)

        # Print and log the results
        toolkit.console(f"Average Test Loss: {avg_test_loss}")
        toolkit.console(f"Average Test Accuracy: {avg_test_accuracy}")

        # Print classification report
        toolkit.console("Classification Report:")
        toolkit.console(classification_report(actual_labels_list, pred_labels_list))

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
        toolkit.console("\nStarted testing...")

        # Evaluate the model on test data
        test_loss, test_accuracy = self.model.evaluate([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']], np.array(y_test))

        pred_logits = self.model.predict([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])
        pred_labels = tf.argmax(pred_logits, axis=1).numpy()  # Use argmax along the appropriate axis to get predicted labels

        labels = {1: 'Positive', 0: 'Negative'}

        # Map the predicted labels to the actual labels
        pred_labels = [labels[i] for i in pred_labels]
        actual_labels = [labels[i] for i in y_test]

        # Print and log the results
        toolkit.console(f"Test Loss: {test_loss}")
        toolkit.console(f"Test Accuracy: {test_accuracy}")
        toolkit.console(classification_report(actual_labels, pred_labels))
        toolkit.console(f"Finished testing in {round(time.time()-t)} seconds.")

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
        toolkit.console("\nStarted predicting...")
        
        confidence_threshold = toolkit.get_config('confidence_threshold')

        input_ids, token_type_ids, attention_mask = self.tokenise(text).values()
        prediction = self.model.predict([np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)])

        labels = {1: 'Positive', 0: 'Negative'}

        pred_logits = prediction.logits
        pred_probs = tf.nn.softmax(pred_logits, axis=1).numpy()
        pred_confidence = np.max(pred_probs, axis=1)

        pred_label_indices = np.argmax(pred_logits, axis=1)
        pred_labels = [labels[i] if confidence >= confidence_threshold else None for i, confidence in zip(pred_label_indices, pred_confidence)]

        truncated_texts = [txt[:64] + '...' if len(txt) > 64 else txt for txt in text] # Truncate texts for logging
        # Log the prediction and confidence in the console
        for txt, label, confidence in zip(truncated_texts, pred_labels, pred_confidence):
            confidence_percent = confidence * 100
            toolkit.console(f"{txt:<67} predicted {label.upper() if label is not None else 'NONE'} with a confidence of {confidence_percent:.2f}%")

        toolkit.console(f"Finished predicting in {round(time.time()-t)} seconds.")

        return pred_labels # Return the labels
