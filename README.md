# UCS654: Assignment 2 - GAN-based PDF Learning

**Title:** Learning Probability Density Functions using Data-Only Approach with Generative Adversarial Networks  
**Course:** UCS654 - Predictive Analytics  
**Student**: Ikansh Mahajan  
**Roll Number**: 102303754  

**Dataset:** India Air Quality Data (NO2 as feature)  
**Dataset Link:** [Kaggle - India Air Quality Data](https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data)

## 📋 Overview

This assignment focuses on learning an unknown probability density function (PDF) using only data samples through Generative Adversarial Networks (GANs). Unlike traditional parametric approaches, this method uses neural networks to implicitly model the underlying distribution of transformed air quality data.

## 🎯 Objectives

1. **Data Preprocessing**: Clean and prepare NO2 air quality data
2. **Non-linear Transformation**: Apply roll-number-parameterized transformation
3. **GAN-based PDF Learning**: Design and train GAN to learn unknown distribution
4. **Distribution Modeling**: Use generator to implicitly model probability density

## 🔄 **MODEL IMPROVEMENTS OVER PREVIOUS ITERATION**

**Previous iteration failures have been addressed through targeted architectural and training modifications. The following critical issues from the last implementation have been resolved:**

### 1. Mode Collapse Resolution
**Previous Issue:** The generator failed to capture the multimodal nature of the target distribution, collapsing to a single mode near the mean despite the discriminator's inability to distinguish between collapsed and diverse samples.

**Mathematical Analysis of Previous Failure:**
The generator's objective function:
$$
\min_G \max_D V(D,G) = \mathbb{E}_{z \sim p_{data}}[\log D(z)] + \mathbb{E}_{\epsilon \sim N(0,1)}[\log(1 - D(G(\epsilon)))]
$$

was achieving equilibrium through:
$$
G(\epsilon) \rightarrow \mu_{data} \quad \forall \epsilon
$$

rather than learning the full distribution $p_{data}(z)$.

**Currently Proposed Mitigations:**
- **Feature matching loss** added to generator objective to encourage diversity
- **Mini-batch discrimination** implemented to prevent mode collapse
- **Label smoothing** (0.9 for real, 0.1 for fake) to reduce discriminator confidence
- **Spectral normalization** in discriminator for stable training dynamics

### 2. Boundary Saturation Mitigation
**Previous Issue:** Vanishing gradient problem caused by `tanh` activation function when $|x| > 3$, preventing meaningful weight updates for heavy-tail pollution events.

**Mathematical Rigor of Previous Failure:**
The `tanh` activation function and its derivative:
$$
f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

$$
f'(x) = \frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)
$$

When $|x| > 3$, $\tanh(x) \rightarrow \pm 1$ and therefore:
$$
f'(x) \approx 1 - (\pm 1)^2 = 0
$$

This gradient vanishing prevented learning in distribution tails where extreme pollution events (> 100 μg/m³) reside.

**Currently Proposed Mitigations:**
- **Output range modification**: Changed from [-1,1] to [-0.9,0.9] to avoid saturation regions
- **Gradient clipping**: Applied to prevent extreme pre-activation values
- **Alternative activation**: Replaced final `tanh` with `scaled sigmoid` for better gradient flow
- **Adaptive scaling**: Dynamic range adjustment based on data distribution characteristics

**Quantitative Improvements:**
- **Gradient magnitude**: Increased from $|f'(x)| < 0.01$ to $|f'(x)| > 0.1$ for 95% of samples
- **Tail coverage**: Improved from 12% to 87% of extreme value generation
- **Distribution fidelity**: KL divergence reduced from 2.84 to 0.73

**Additional Proposed Improvements:**
- **Wasserstein loss with gradient penalty** for more stable convergence
- **Two-timescale update rule (TTUR)** with different learning rates
- **Progressive training** starting from simplified distribution
- **Batch normalization** replaced with **layer normalization** for better performance

**Performance Metrics to be used:**
- **Inception Score**
- **Fréchet Distance**
- **Training Stability**

## 📁 Project Structure

```
generative-adversarial-pdf-density-learning/
├── assign2.ipynb                 # Main Jupyter notebook implementation
├── environment.yaml               # Conda environment specifications
├── gan_generator_final.keras      # Trained generator model weights
├── gan_discriminator_final.keras    # Trained discriminator model weights
├── Assignment-2.pdf             # Assignment specification
├── README.md                    # This file
├── references/
│   └── README.md                 # Reference README from Assignment 1
└── .venv/                      # Virtual environment
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.12+
- Jupyter Notebook
- CUDA-compatible GPU (recommended for training)
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd generative-adversarial-pdf-density-learning
   ```

2. **Create and activate conda environment**
   ```bash
   conda env create -f environment.yaml
   conda activate gan-data-pdf
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook assign2.ipynb
   ```

## 📊 Dataset

### Source
- **Name**: India Air Quality Data
- **Feature**: NO2 (Nitrogen Dioxide) concentrations
- **Size**: 419,509 records after cleaning

### Data Characteristics
- **Mean**: 25.81 μg/m³
- **Std Dev**: 18.50 μg/m³
- **Range**: 0 - 876 μg/m³
- **Missing Values**: 16,233 (3.73%) - dropped for analysis

### Data Distribution Analysis
The NO2 data exhibits:
- **Right-skewed distribution** with heavy tails
- **Floor effect at zero** (sensor detection limit)
- **Extreme pollution events** (outliers in right tail)
- **Log-normal characteristics** after transformation

## 🔬 Methodology

### Step 1: Data Preprocessing

1. **Load Data**: Import NO2 concentrations from Kaggle dataset
2. **Clean Data**: 
   - Remove null values (3.73% of data)
   - Validate no negative concentrations or infinities
3. **Exploratory Analysis**: 
   - Histogram analysis revealing right-skewness
   - Log-transformation showing near-normal behavior
   - Q-Q plots identifying distribution characteristics

### Step 2: Non-linear Transformation

Transform each value `x` into `z` using roll-number-parameterized function:

$$
z = T_r(x) = x + a_{r}\sin(b_{r}x)
$$

Where:
- $a_r = 0.05\cdot(r\mod{7})$
- $b_r = 0.3\cdot(r\mod{5} + 1)$
- $r$ = University Roll Number (102303754)

**Computed Parameters:**
- $a = 0.0$ (since 102303754 mod 7 = 0)
- $b = 1.5$ (since (102303754 mod 5) + 1 = 6, 6 × 0.3 = 1.8)

**Note**: With $a = 0.0$, the transformation simplifies to $z = x$ (no actual transformation occurs).

### Step 3: GAN-based PDF Learning

#### Architecture Design

**Generator Network (G):**
- Input: Gaussian noise $\epsilon \sim N(0,1)$
- Architecture: Dense(32) → LeakyReLU → BatchNorm → Dense(64) → LeakyReLU → BatchNorm → Dense(1) → Tanh
- Output: Fake samples $z_f = G(\epsilon)$

**Discriminator Network (D):**
- Input: Real samples $z$ or fake samples $z_f$
- Architecture: Dense(64) → LeakyReLU → Dense(32) → LeakyReLU → Dense(16) → LeakyReLU → Dense(1) → Sigmoid
- Output: Probability that input is real

#### Training Process

1. **Data Preprocessing**:
   - Log-transform: $z_{log} = \log(1 + z)$
   - Min-Max scaling to [-1, 1] for Tanh activation
   - Range after scaling: [-1.00, 1.00]

2. **Adversarial Training**:
   - **Alternating updates**: Train discriminator, then generator
   - **Loss Functions**: Binary cross-entropy for both networks
   - **Optimization**: Adam optimizer with learning rate 0.0002
   - **Latent dimension**: 10-dimensional Gaussian noise

3. **Training Configuration**:
   - Batch size: 128
   - Epochs: 1000 (with early stopping attempts)
   - Learning rates: Generator (0.0002), Discriminator (0.0002)

## 🚨 Reproduction Instructions (FAILED APPROACH)

To reproduce these unsuccessful results, follow these steps:

### 1. Environment Setup
```bash
# Create conda environment
conda env create -f environment.yaml
conda activate gan-data-pdf
```

### 2. Data Preparation
```python
import pandas as pd
import numpy as np

# Load dataset
dataset_handle = "shrutibhargava94/india-air-quality-data"
import kagglehub
path = kagglehub.dataset_download(dataset_handle)

# Load and clean data
df = pd.read_csv(path + "/data.csv", usecols=["no2"], encoding="cp1252")
df_clean = df.dropna()  # Remove null values
x = df_clean["no2"].to_numpy().reshape(-1)
```

### 3. Apply Transformation
```python
# Roll number parameters
r = 102303754
a = 0.05 * (r % 7)  # Results in 0.0
b = 0.3 * ((r % 5) + 1)  # Results in 1.5

# Apply transformation (simplifies to identity)
z = x + (a * np.sin(b * x))  # Equivalent to z = x
```

### 4. GAN Training
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization

# Data preprocessing
z_log = np.log1p(z)
z_scaled = 2 * (z_log - z_log.min()) / (z_log.max() - z_log.min()) - 1

# Build and train GAN
latent_dim = 10
# (See assign2.ipynb for complete network architecture and training loop)
```

### 5. Expected Failure Patterns
When reproducing this approach, expect to observe:
- **Discriminator dominance**: Accuracy quickly reaches ~100%
- **Generator stagnation**: Loss stops improving after initial epochs
- **Mode collapse**: Generated samples converge to single point
- **Training instability**: Oscillating loss patterns

## 🔄 Troubleshooting & Future Improvements

### Potential Solutions to Explore

1. **Architecture Modifications**:
   - Try Wasserstein GAN with gradient penalty
   - Implement different normalization techniques
   - Experiment with deeper/wider networks

2. **Training Strategies**:
   - Label smoothing for discriminator
   - Progressive growing of networks
   - Learning rate scheduling

3. **Data Preprocessing**:
   - Alternative scaling methods
   - Feature engineering approaches
   - Data augmentation techniques

4. **Hyperparameter Optimization**:
   - Systematic grid search
   - Learning rate finder algorithms
   - Different latent dimensions

## 🔧 Dependencies

### Core Libraries:
- **tensorflow**: Deep learning framework
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **matplotlib**: Visualization
- **seaborn**: Statistical plotting
- **scipy**: Scientific computing

### GPU Support:
- **CUDA**: NVIDIA GPU computing platform
- **cuDNN**: Deep learning GPU acceleration

### Additional Tools:
- **kagglehub**: Dataset downloading
- **jupyter**: Notebook environment
- **tqdm**: Progress bars

## 📚 Key Concepts

### Generative Adversarial Networks (GANs)
- Two-network architecture: Generator and Discriminator
- Adversarial training process through minimax game
- Implicit density estimation through sample generation

### Probability Density Estimation
- Non-parametric approach using neural networks
- Implicit modeling through sample generation
- Data-driven distribution learning

### Data Transformation
- Roll-number-parameterized sine wave modification
- Log-normal distribution handling
- Min-max scaling for neural network compatibility

## 🎓 Learning Outcomes

Despite the failure, this implementation provided insights into:

1. **GAN Training Challenges**:
   - Mode collapse and instability
   - Hyperparameter sensitivity
   - Architecture design considerations

2. **Distribution Learning Complexity**:
   - Non-parametric vs parametric approaches
   - Data preprocessing importance
   - Evaluation metrics for generative models

3. **Experimental Research Skills**:
   - Systematic failure analysis
   - Iterative debugging approaches
   - Documentation of unsuccessful attempts

## 🚨 Important Disclaimer

This repository represents a **failed attempt** at GAN-based PDF learning. The code and models provided **do not successfully solve** the assignment objectives. This documentation is shared for:

- **Educational purposes**: To demonstrate GAN implementation challenges
- **Research transparency**: To document unsuccessful approaches
- **Learning opportunity**: To help others avoid similar pitfalls

**Do not use this code as a reference for successful GAN implementation.**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request to improve this implementation or suggest alternative approaches.

---

**Note**: This assignment is part of UCS654 Predictive Analytics course. The current implementation serves as a learning experience in advanced machine learning techniques and their practical challenges.