# Disease Prediction

## Overview
This project aims to predict diseases based on symptoms using Machine Learning. The dataset contains various symptoms and their corresponding diseases, which are used to train a model for accurate predictions.

## Features
- Predict diseases based on user-input symptoms.
- Uses machine learning algorithms for prediction.
- Supports CSV datasets for training and testing.
- User-friendly interface for ease of use.

## Dataset
- **File:** `Final_Augmented_dataset_Diseases_and_Symptoms.csv`
- **Size:** ~182MB
- **Source:** Kaggle
  
## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Pandas
- NumPy
- Scikit-learn

### Clone the Repository
```bash
git clone https://github.com/Hardik450/disease_prediction.git
cd disease_prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Handling Large Dataset
Since GitHub restricts files over **100MB**, the dataset is stored on Google Drive.

#### Download the dataset:
```python
import gdown
file_id = "1S4YrSFcYQ0T2NQhl0L3hK-jP9PBmqqWS"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "Final_Augmented_dataset_Diseases_and_Symptoms.csv", quiet=False)
```

## Usage
### Write the correct path of the file
- If you have imported in same directory use:
  ```python
  data = pd.readcsv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
  ```
  
### Run your file in Google Colab
```bash
python disease_prediction.py
```


## Contributing
Feel free to contribute! Fork the repository and create a pull request with your improvements.

## License
This project is licensed under the MIT License.

---
📌 **Author:** Hardik450  
📧 **Contact:** hardikjainharsora@gmail.com

