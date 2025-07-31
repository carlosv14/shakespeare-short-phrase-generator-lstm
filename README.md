# Shakespeare Text Generator

This project is a character-level text generator trained on Shakespeare's works. It uses an LSTM-based neural network to generate text based on a given prompt.

## Features

- Train a character-level LSTM model on Shakespeare's text.
- Generate text based on a user-provided prompt.
- Adjustable parameters for temperature, top-k sampling, and maximum words.
- Web interface built with Flask for easy interaction.

## Project Structure

```
.
├── app.py               # Flask application entry point
├── data.py              # Data preprocessing utilities
├── dependencies.txt     # Project dependencies
├── model.py             # LSTM model implementation
├── predict.py           # Text generation logic
├── train.py             # Model training script
├── data/
│   └── shakespeare.txt  # Training data (Shakespeare's works)
├── templates/
│   └── index.html       # HTML template for the web interface
```

## Requirements

Install the required dependencies using the following command:

```bash
pip install -r dependencies.txt
```

Dependencies:
- Flask
- PyTorch
- Unidecode
- Numpy

## Usage

### Training the Model

To train the model, run the `train.py` script:

```bash
python train.py data/shakespeare.txt
```

| Parameter          | Default Value | Description                                      |
|--------------------|---------------|--------------------------------------------------|
| `--example-length` | `200`         | Length of each training example in characters.   |
| `--n-epochs`       | `2000`          | Number of training epochs.                     |
| `--learning-rate`  | `0.005`       | Learning rate for the optimizer.                 |
| `--hidden-size`    | `100`         | Number of hidden units in the LSTM layer.        |
| `--n-layers`       | `2`           | Number of LSTM layers.                           |
| `--print-every`    | `100`         | Frequency of printing training progress.         |

### Running the Web Application

Start the Flask web application by running:

```bash
python app.py
```

The application will be accessible at `http://127.0.0.1:5000`.

### Generating Text

1. Open the web application in your browser.
2. Enter a prompt, temperature, top-k value, and maximum words.
3. Click "Generate" to see the generated text.

## File Descriptions

- **`app.py`**: Handles the Flask web server and routes.
- **`data.py`**: Contains functions for reading and preprocessing the text data.
- **`model.py`**: Defines the LSTM-based text generation model.
- **`predict.py`**: Implements the text generation logic using the trained model.
- **`train.py`**: Script for training the LSTM model on the dataset.
- **`data/shakespeare.txt`**: The dataset used for training, containing Shakespeare's works.
- **`templates/index.html`**: HTML template for the web interface.

## License

This project is for educational purposes and does not include a specific license. Please ensure compliance with any applicable copyright laws when using the Shakespeare dataset.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework.
- Shakespeare's works for the dataset.

## Running the Project with Docker

You can run this project using Docker to simplify the setup process. Follow the steps below:

### Build the Docker Image

First, build the Docker image:

```bash
docker build -t short-phrase-generator .
```

### Run the Docker Container

Run the container and map the application to a port on your host machine (e.g., port 5000):

```bash
docker run -p 5000:5000 short-phrase-generator
```

### What Happens During Startup?

1. **Model Check**:  
   When the container starts, the `entrypoint.sh` script checks if the `short-phrase-generation.pt` file (the trained model) exists in the container.
   
2. **If the Model File Exists**:  
   - The script skips the training process and directly starts the Flask application.

3. **If the Model File Does Not Exist**:  
   - The script automatically trains the model using the `train.py` script with the default dataset (`data/shakespeare.txt`).
   - Once training is complete, the Flask application is started.

### Access the Application

After running the container, you can access the application in your browser at:

```
http://localhost:5000
```

### Example Output

- If the model file exists:
  ```
  Model file found. Skipping training.
  * Running on http://0.0.0.0:5000 (Press CTRL+C to quit)
  ```

- If the model file does not exist:
  ```
  Model file not found. Training the model...
  Training complete. Starting the application...
  * Running on http://0.0.0.0:5000 (Press CTRL+C to quit)
  ```

This setup ensures that the model is trained only when necessary, making it easy to deploy and run the application in any environment.

