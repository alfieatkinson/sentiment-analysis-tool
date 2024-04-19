import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

import toolkit

# Load and preprocess the data
def initialise_dataset():
    t = time.time() # Start timer
    processed_text = [] # Define list of processed text

    # Importing the dataset
    DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    dataset = pd.read_csv((toolkit.get_dir() + '/datasets/sentiment140/dataset.csv'), encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

    dataset = dataset[['sentiment', 'text']] # Removing the unnecessary columns
    dataset['sentiment'] = dataset['sentiment'].replace(4, 1) # Replacing the values to ease understanding
    texts, sentiment = list(dataset['text']), list(dataset['sentiment']) # Storing data in lists.

    # Clean each text
    for text in texts:
        processed_text.append(toolkit.data.preprocess.clean(text))

    print(f'Text Preprocessing complete.')
    print(f'Time Taken: {round(time.time()-t)} seconds') # End timer

    return processed_text, sentiment

# Save the tokeniser and model
def save_model(tokeniser, model):
    tokeniser.save_pretrained(toolkit.get_dir() + "/models/tokeniser") # Save tokeniser
    model.save_pretrained(toolkit.get_dir() + "/models/model") # Save model

# Load the tokeniser and model
def load_model():
    tokeniser = BertTokenizer.from_pretrained(toolkit.get_dir() + "/models/tokeniser") # Load tokeniser
    model = TFBertForSequenceClassification.from_pretrained(toolkit.get_dir() + "/models/model") # Load model

    return tokeniser, model

# Split and train the model
def train():
    t = time.time() # Start timer
    print(f"Start time: {toolkit.time()}")

    text, sentiment = initialise_dataset()

    # Split the data into 95% training and 5% testing
    X_train, X_test, y_train, y_test = train_test_split(text, sentiment, test_size = 0.05, stratify=sentiment, random_state = 0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=0)
    print()
    print("Data split done.")

    # Tokenise and encode the data using BERT tokeniser
    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    max_len= 128
    # Tokenise and encode the sentences
    X_train_encoded = tokeniser.batch_encode_plus(X_train, padding=True, truncation=True, max_length = max_len, return_tensors='tf')
    X_val_encoded = tokeniser.batch_encode_plus(X_val, padding=True, truncation=True, max_length = max_len, return_tensors='tf')
    X_test_encoded = tokeniser.batch_encode_plus(X_test, padding=True, truncation=True, max_length = max_len, return_tensors='tf')

    # Check the encoded dataset
    k = 0
    print()
    print('Training Comments: ', X_train[k])
    print('\nInput Ids: \n', X_train_encoded['input_ids'][k])
    print('\nDecoded Ids: \n', tokeniser.decode(X_train_encoded['input_ids'][k]))
    print('\nAttention Mask: \n', X_train_encoded['attention_mask'][k])
    print('\nLabels: ', y_train[k])

    # Initialise the model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Compile the model with optimiser, loss function, and metrics
    optimiser = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimiser, loss=loss, metrics=[metric])

    # Train the model
    history = model.fit([np.array(X_train_encoded['input_ids']), np.array(X_train_encoded['token_type_ids']), np.array(X_train_encoded['attention_mask'])], np.array(y_train), validation_data=([np.array(X_val_encoded['input_ids']), np.array(X_val_encoded['token_type_ids']), np.array(X_val_encoded['attention_mask'])], np.array(y_val)), batch_size=32, epochs=3)

    print()
    print(f'Training complete.')
    print(f"Finish time: {toolkit.time()}") # End timer
    print(f'Time Taken: {round(time.time()-t)} seconds')

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate([np.array(X_test_encoded['input_ids']), np.array(X_test_encoded['token_type_ids']), np.array(X_test_encoded['attention_mask'])], np.array(y_test))

    print()
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")
    print("Evaluation complete.")

    save_model(tokeniser, model) # Save the model

    pred = model.predict([np.array(X_test_encoded['input_ids']), np.array(X_test_encoded['token_type_ids']), np.array(X_test_encoded['attention_mask'])])
    logits = pred.logits

    pred_labels = tf.argmax(logits, axis=1) # Use argmax along the appropriate axis to get predicted labels
    pred_labels = pred_labels.numpy() # Convert the predicted labels to a numpy array

    label = {
        1: 'Positive',
        0: 'Negative'
    }

    # Map the predicted labels to the actual labels
    pred_labels = [label[i] for i in pred_labels]
    actual_labels = [label[i] for i in y_test]

    print()
    print(f"Predicted labels: {pred_labels[:10]}")
    print(f"Actual labels: {actual_labels[:10]}")
    print()
    print("Classification report: \n", classification_report(actual_labels, pred_labels))
    print("Testing complete.")

def predict(text):
    t = time.time() # Start timer
    print()
    print(f"Start time: {toolkit.time()}")

    tokeniser, model = load_model() # Load trained model
    text = toolkit.data.preprocess.clean(text) # Clean the text

    # Convert text to a list if it is not already a list
    if not isinstance(text, list):
        text = [text]

    input_ids, token_type_ids, attention_mask = tokeniser.batch_encode_plus(text, padding=True, truncation=True, max_length=128, return_tensors='tf').values()
    prediction = model.predict([np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)])

    label = {
        1: 'Positive',
        0: 'Negative'
    }

    pred_labels = tf.argmax(prediction.logits, axis=1) # Use argmax along the appropriate axis to get the predicted labels
    pred_labels = [label[i] for i in pred_labels.numpy().tolist()] # Convert the TensorFlow tensor to a NumPy array and then to a list to get predicted sentiment labels

    print(f"Finish time: {toolkit.time()}") # End timer
    print(f'Time Taken: {round(time.time()-t)} seconds')

    return pred_labels