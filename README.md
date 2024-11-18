# Federated Learning with Flower and CelebA Dataset

This project demonstrates a federated learning setup using the [Flower](https://flower.dev/) framework and PyTorch with the CelebA dataset. The experiment involves both IID and non-IID data splits and uses a pre-trained MobileNetV2 model for efficient training.

---

## Features

- **Data Partitioning**: Generate IID and non-IID distributions for federated training .  
- **Pre-trained Model**: Leverage MobileNetV2 with a frozen feature extractor for computational efficiency.  
- **Federated Learning Simulation**: Simulate federated training with 50 clients over 10 communication rounds across demographic groups.  
- **Evaluation**: Perform analysis using a classification report, confusion matrix and learning curves.  
- **Real-World Federated Learning Execution**: Implement federated learning in a real-world setting using gRPC for seamless client-server communication, replicating practical deployment scenarios.  

---
## Requirements

This project requires Python 3.6 or higher. The complete list of dependencies is in `requirements.txt`.

### Installation

1. Install Python 3.6 or higher.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
    ```
---
## Project Structure
```bash
federated-learning-celeba/
├── configs/
│   ├── client_id.py
│   ├── config.py
├── data/
│   ├── data_loader.py
│   ├── fed_dataset.py
│   └── partitioner/
│       ├── visual_dirichlet_partitioner.py
│       └── iid_partitioner.py
├── model/
│   └── mobilenetv2.py
├── results/
│   ├── server/
├── strategies/
│   ├── FedAvg.py
├── client.py
├── main.py
├── README.md
├── requirements.txt
├── server.py
└── task.py
```

### Key Components

- **`configs/`**: Configuration files to manage both global and client-specific settings.
  - **`client_id.py`**: Defines unique identifiers and configurations for each client.
  - **`config.py`**: Centralized configuration file for global settings.
- **`data/`**: Scripts and utilities for data loading, preprocessing, and partitioning.
  - **`data_loader.py`**: Handles the loading and preprocessing of the CelebA dataset.
  - **`fed_dataset.py`**: Defines the `FedDataset` class for creating federated datasets.
  - **`partitioner/`**: Implements strategies for data partitioning.
    - **`visual_dirichlet_partitioner.py`**: Creates non-IID distributions using a Dirichlet distribution.
    - **`iid_partitioner.py`**: Creates IID partitions for federated learning experiments.
- **`model/`**: Contains the definition of the training model.
  - **`mobilenetv2.py`**: Implements a MobileNetV2 model with a frozen feature extractor and customizable classifier head.
- **`results/`**: Stores results and logs from experiments.
  - **`server/`**: Contains server-side evaluation results and metrics.
- **`strategies/`**: Includes custom federated learning strategies.
  - **`FedAvg.py`**: Implementation of the Federated Averaging algorithm.
- **`client.py`**: Defines the behavior of federated clients, including local training and communication with the server.
- **`server.py`**: Manages the federated learning server, including aggregation, coordination, and evaluation.
- **`task.py`**: Contains task-specific logic for model training, testing, and evaluation.
- **`main.py`**: Central script to orchestrate and execute the federated learning experiment.
- **`requirements.txt`**: Lists all dependencies required to run the project.
- **`README.md`**: Provides documentation, including project overview, usage instructions, and descriptions of components.

---
## **Implementation Details**

### Step 1: Dataset Loading and Partitioning

The first step in the project is to load the CelebA dataset, add demographic labels, and partition the dataset into client-specific subsets for federated learning. This step allows flexible data distribution strategies, such as IID and non-IID, to simulate diverse real-world scenarios.

1. **Load the CelebA Dataset**  
   The CelebA dataset is loaded using the `datasets` library, making it easily accessible for partitioning and labeling.

2. **Add Demographic Labels**  
   A custom function, `add_demographic_labels`, categorizes each data point based on two attributes: `Male` and `Young`. The demographic labels are:
   - **0**: Not Male & Not Young
   - **1**: Not Male & Young
   - **2**: Male & Not Young
   - **3**: Male & Young

   These labels enable the analysis of model performance across demographic groups.

3. **Select Partitioning Strategy**  
   The dataset is partitioned based on the specified `distribution`:
   - **IID Partitioning**: Evenly splits the dataset across all clients.
   - **Non-IID Partitioning**: Uses the **Visual Dirichlet Partitioner** to create skewed distributions based on demographic labels, with parameters such as `alpha` controlling non-uniformity.

4. **Partition the Dataset**  
   The dataset is divided into `num_partitions`, and the specified `partition_id` determines the data subset for a client using `FedDataset` class.

5. **Split into Training and Validation**  
   Each client’s data subset is split into 80% training and 20% validation data.

6. **Apply Transformations**  
   PyTorch transformations, such as normalization, are applied to prepare the data for training.

7. **Create DataLoaders**  
   The function returns three DataLoader objects:
   - **`trainloader`**: For client-specific training data.
   - **`valloader`**: For local validation.
   - **`testloader`**: For global testing across all clients.


### **Step 2: Model Definition**
The `mobilenetv2.py` script defines a MobileNetV2 model:
- **Feature Extractor**: Pre-trained and frozen to reduce computational overhead.
- **Classifier Head**: Customizable for the specific task of classifying CelebA attributes.

### **Step 3: Federated Learning Setup**
This step involves setting up the core components of the federated learning framework. It defines the interactions between clients and the server, implements the Federated Averaging strategy, and integrates task-specific logic for model training and evaluation.


#### **Client Implementation (`client.py`)**

The client script handles the following key responsibilities:
- **Local Training**: Each client trains its model on its partitioned dataset, leveraging the task-specific training and testing functions.
- **Communication**: The client sends its locally trained model updates (weights) to the server and receives the global model from the server after aggregation, and using this model for local training.


#### **Server Implementation (`server.py`)**

The server script manages the federated learning process by:
- **Model Aggregation**: Uses the Federated Averaging strategy to aggregate model updates from all clients.
- **Coordination**: Orchestrates communication rounds, ensuring proper synchronization between clients and the global model.
- **Evaluation**: Includes evaluation function to allow the server to assess the aggregated model’s performance on a global test set after each communication round.


#### **Strategy (`FedAvg.py`)**

The Federated Averaging strategy script implements the core aggregation logic. Key features:
- **Model Aggregation**: Combines model weights from clients based on their dataset sizes.
- **Modified Evaluation Function**: Enhances the standard FedAvg algorithm by enabling the server to evaluate the model’s performance after every communication round, providing real-time feedback on convergence.



#### **Task-Specific Logic (`task.py`)**

This script contains essential functions for training, testing, and evaluating the model, including:
- **Training (`train_fn`)**: Handles local model training on client devices.
- **Testing (`test_fn`)**: Evaluates the model on the global test set and client-specific validation sets.
- **Metrics Calculation**: Computes and returns metrics such as accuracy, precision, recall, and F1-score. Results from each round are saved in the `results/` directory for analysis.



### **Step 4: Running the Experiment**

The `main.py` script serves as the entry point to orchestrate the entire federated learning experiment. It ensures proper coordination between the server and clients while handling the configuration of the training process and evaluation.

1. **Setting Up the Server and Clients**  
   The script initializes:
   - The **server application** (`ServerApp`) using the `server_fn` defined in `server.py`, which manages global model aggregation and evaluation.
   - The **client application** (`ClientApp`) using the `client_fn` defined in `client.py`, which handles local training and communication with the server.

2. **Configuring the Number of Communication Rounds**  
   The number of participating clients (`num_supernodes`) and backend settings are loaded dynamically from the configuration file (`configs/config.py`).

3. **Initiating Training and Evaluation**  
   The `run_simulation` function coordinates the communication rounds between the server and clients:
   - Starts the training process across multiple clients.
   - Aggregates local model updates at the server.
   - Evaluates the global model after each communication round.

### **Step 5: Evaluation**

The evaluation process assesses the performance of the global model after each communication round. Results are saved in the `results/` directory for detailed analysis and visualization.

#### **Saved Evaluation Metrics**
- **Classification Report**:  
  Includes precision, recall, and F1-score for each class, providing insights into the model's performance across different demographic labels.
  
- **Confusion Matrix**:  
  Captures the distribution of correct and incorrect predictions, helping identify patterns and areas of improvement.

- **Learning Curves**:  
  Tracks training and validation accuracy and loss over communication rounds, providing a clear view of the model’s convergence behavior.

#### **Detailed Analysis**
A Jupyter notebook, `Federated_Learning_with_Flower_and_CelebA.ipynb`, is included for in-depth evaluation and visualization. This notebook provides:
- Visual representations of learning curves.
- Insights into class-wise performance using classification reports and confusion matrices.
- A summary of key metrics over the federated learning process.

---

## **Usage Instructions**

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/federated-learning-celeba.git
   cd federated-learning-celeba
    ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
    ```
3. Configure the simulation:

    Edit the configs/config.py file to set up the desired number of clients (NUM_PARTITIONS), backend configurations (backend_config), and other parameters.

4. Run the federated learning experiment:
   ```bash
   python main.py
    ```


---
## **Results and Analysis**
- Explore the saved evaluation metrics in the `results/server/` directory.
- Use the Jupyter notebook `Federated_Learning_with_Flower_and_CelebA.ipynb` for detailed analysis and visualization.


---
## **License**  
This project is licensed under the MIT License - see the LICENSE file for details.

---
## **Acknowledgements**
- Flower: https://flower.dev/
- PyTorch: https://pytorch.org/
- CelebA Dataset from [MMLab](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)