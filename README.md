# Energy Consumption Anomaly Detection with Contrastive Learning

This project aims to detect anomalies in energy consumption data using a contrastive learning approach. The model is trained to learn representations of time series data and identify unusual patterns that may indicate anomalies.

## Project Structure

- `constrastive.py`: Contains the main code for data preprocessing, model definition, training, and anomaly detection.
- `dataset/maio.csv`: The dataset used for training and testing the model.
- `anomaly_plot.png`: A plot showing the detected anomalies in the energy consumption data.

## Requirements

- Python 3.x
- PyTorch
- Pandas
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/romoreira/energy-consumption-anomaly-detection.git
    cd energy-consumption-anomaly-detection
    ```

2. Install the required packages:
    ```sh
    pip install torch pandas matplotlib scikit-learn
    ```

## Usage

1. Prepare the dataset:
    - Place your CSV file in the `dataset` directory.
    - Update the file path in `constrastive.py` if necessary.

2. Run the training script:
    ```sh
    python constrastive.py
    ```

3. The script will output the training loss for each epoch and generate a plot (`anomaly_plot.png`) showing the detected anomalies.

## Methodology

- **Data Preprocessing**: The script reads the CSV file, normalizes the `corriente` feature, and creates time series pairs for training.
- **Model**: A Siamese network with an LSTM encoder is used to learn representations of the time series data.
- **Loss Function**: The InfoNCE loss is used to train the model in a contrastive manner.
- **Anomaly Detection**: After training, the model is used to compute distances between time series pairs, and anomalies are identified based on a threshold.

## Results

The detected anomalies are plotted and saved as `anomaly_plot.png`. The plot shows the energy consumption over time with anomalies marked in red.

## License

This project is licensed under the MIT License.