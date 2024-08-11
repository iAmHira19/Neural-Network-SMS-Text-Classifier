# Neural-Network-SMS-Text-Classifier



```markdown
# SMS Spam Classifier

## Overview

This project involves creating a machine learning model to classify SMS messages as either "ham" (normal) or "spam" (advertisement). The classification is performed using a Naive Bayes model with text data processing.

## Dataset

The dataset used for this project is the SMS Spam Collection dataset. It contains SMS messages classified into two categories: "ham" and "spam".

## Getting Started

To get started with this project, follow these instructions:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. **Install Dependencies**

   Make sure you have `pandas`, `numpy`, `scikit-learn`, and `matplotlib` installed. You can install the dependencies using:

   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. **Download the Dataset**

   Place the dataset file `spam.csv` in the `/content/` directory or update the path in the code accordingly.

4. **Run the Notebook**

   Open `spam_classifier.ipynb` in Google Colaboratory or Jupyter Notebook and execute the cells.

## Usage

To classify a message as "ham" or "spam", use the `predict_message` function:

```python
def predict_message(message):
    # Predict the class and probability of the message
    prediction = model.predict([message])[0]
    likelihood = model.predict_proba([message])[0][1]
    label = 'spam' if prediction == 1 else 'ham'
    return [likelihood, label]
```

**Example:**

```python
test_message = "Congratulations! You've won a $1000 gift card. Call now to claim your prize!"
result = predict_message(test_message)
print(result)
```

The function returns a list where the first element is the likelihood of the message being spam, and the second element is the predicted label.

## Evaluation

The model is evaluated using various metrics including precision, recall, and F1-score. The classification report provides detailed performance metrics.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The SMS Spam Collection dataset is available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

## Contact

For any questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com).
```

### Tips for Customization

- **Repository URL**: Replace `https://github.com/yourusername/sms-spam-classifier.git` with the actual URL of your GitHub repository.
- **Dataset Path**: Ensure that the path to the dataset file in the code is correctly specified based on where you place the dataset.
- **Contact Information**: Update the contact section with your email or preferred contact method.

