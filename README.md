## Intrusion Detection System
This is a simple Intrusion Detection System (IDS) that uses machine learning models such as XGBoost and Ensemble models to detect malicious traffic. The dataset used is the [NSL-KDD dataset](https://www.unb.ca/cic/datasets/nsl.html) which is a modified version of the KDD dataset. The dataset is available in the [data](data) folder. The dataset is also available on [Kaggle](https://www.kaggle.com/akashkr/nslkdd).

## Installation and Setup
Create a virtual environment and install the requirements using the following command:
```sh conda create --prefix ./env python=3.8```

Start the API server using the following command:
```sh python app.py```

To run the API server in debug mode, use the following command:
```sh python app.py --debug=True```