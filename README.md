# Synthetic Residential Water Usage Generation via Diffusion Models

## Table of Contents
- [The Problem](#the-problem)
- [The Goal](#the-goal)
- [State of the Art](#state-of-the-art)
- [Data Privacy Notice](#data-privacy-notice)
- [License](#license)
- [Model Architecture](#model-architecture)
- [Future Directions & Proof of Concept](#future-directions--proof-of-concept)
  - [1. Data Pipeline & Feature Engineering](#1-data-pipeline--feature-engineering)
  - [2. Zero-Inflated Dual-Track Representation](#2-zero-inflated-dual-track-representation)
  - [3. Network Architecture](#3-network-architecture)
  - [4. Diffusion Mathematics](#4-diffusion-mathematics)

## The Problem
One of the major bottlenecks in developing water disaggregation methods for flow trace analysis is the lack of publicly available data. Household water consumption data is highly sensitive information, making it extremely difficult for researchers to access real-world datasets required for analysis and method development. 

There is a significant need for a robust tool capable of simulating realistic residential water usage data. Such a tool would allow researchers to freely develop, test, and benchmark methods for water end-use disaggregation without relying on restricted or private datasets.

## The Goal
This repository serves as a proof-of-concept for using a **1D Conditional Diffusion Model** to generate highly realistic, synthetic residential water usage data. By modeling the complex diurnal patterns and weather-driven irrigation events of individual households, we aim to provide a foundation for a generative tool that researchers can use for end-use disaggregation.

![Synthetic Samples](images/synthetic_samples_poc.png)

## State of the Art
Currently, the state-of-the-art simulator for this domain is [STREaM](https://github.com/acominola/STREaM) by Cominola et al. (2016). Our approach explores modern deep generative models (specifically, Denoising Diffusion Probabilistic Models) as a novel alternative to create similar simulation tools.

## Data Privacy Notice
**For security and privacy reasons, none of the real-world AMI data used to train the model is included in this repository.** However, all of the code required to build the features, process the windows, train the model, and generate the synthetic samples is provided here.

## License
This project is licensed under the [Apache License 2.0](LICENSE). This means the code is fully open-source and includes an explicit patent grant, but it is provided "as is" without any warranties, and the creators cannot be held legally liable for how the software is used.

---

## Model Architecture

This section details the complete end-to-end architecture of the 1D Conditional Diffusion Model designed for synthetic residential water usage generation. It covers the data processing pipeline, the dual-track mathematical solutions for zero-inflated data, the dual-stream network architecture, and the fundamental diffusion equations.

### 1. Data Pipeline & Feature Engineering

The model operates on high-resolution Advanced Metering Infrastructure (AMI) data, sampled at 15-minute intervals (96 intervals per 24-hour period).

#### 1.1 Feature Dictionary
The model is conditioned on two explicit streams of information to separate human diurnal behavior from stochastic climate-driven behavior (like irrigation).

* **Indoor/Behavioral Stream (`c_in`, 12 features)**
  * **Temporal Cyclicals (`month_sin/cos`, `dow_sin/cos`, `hour_sin/cos`)**: Bounded `[-1, 1]` trigonometric encodings that allow the model to learn the cyclical nature of time seamlessly.
  * **Lagged Usage (`usage_15m`, `usage_30m`, `usage_1h`, `usage_6h`, `usage_24h`, `usage_48h`)**: Log-normalized historical usage allowing the model to condition on recent and mid-range household activity states.

* **Outdoor/Climate Stream (`c_out`, 10 features)**
  * **Current Climate**: `temp_c`, `precip_mm`, `snow_cm`, `snow_flag`.
  * **Lagged Climate**: `temp_1h`, `temp_24h`, `temp_48h`.
  * **Accumulated Climate Variables**:
    * `precip_3d`: 3-day rolling sum of precipitation.
    * `snow_24h`: 24-hour rolling sum of snowfall.
    * `gdd_7d`: 7-day rolling sum of Growing Degree Days.

#### 1.2 Rolling Windows (Translation Invariance)
Instead of disjoint daily windows, the pipeline extracts sequences using a **rolling stride of 1 interval**. This provides massive data augmentation and forces the model to learn *Translation Invariance*—understanding what the shape of a toilet flush or shower looks like structurally, regardless of what index it occupies in the 96-step sequence.

### 2. Zero-Inflated Dual-Track Representation

Standard generative models fail on raw water usage because the data is *zero-inflated* and heavily right-skewed. A house uses exactly $0.0$ gallons for the vast majority of the day, but experiences sudden, massive bursts.

To solve this mathematically, the Dataloader applies a **Hurdle Transformation**, splitting the single water usage track into a 2-channel tensor:

1.  **Occurrence Mask ($m$)**: A binary mask `[0.0, 1.0]` indicating if *any* flow occurred.
2.  **Log-Magnitude ($v$)**: A strictly continuous value representing the absolute volume.
    *   $v_{raw} = \log(x + 1)$
    *   $v_{norm} = \frac{v_{raw} - \mu}{\sigma}$

### 3. Network Architecture

The backbone is a **Dual-Stream 1D U-Net** utilizing FiLM (Feature-wise Linear Modulation) conditioning.

![UNet Architecture](images/diffusion_unet_u_shape.png)

```mermaid
graph TD
    X["Noisy Input x_t<br>(2 Channels: Mask, Mag)"] --> U1["UNet_In"]
    X --> U2["UNet_Out"]

    T["Time Embedding<br>t"] -.-> U1
    T -.-> U2

    C_IN["c_in<br>(Behavioral/Time)"] --> E_IN["Dense Encoder"]
    C_OUT["c_out<br>(Climate/Weather)"] --> E_OUT["Dense Encoder"]

    E_IN --> CAT["Concat Full Context"]
    E_OUT --> CAT

    CAT -.->|"FiLM Injection"| U1
    CAT -.->|"FiLM Injection"| U2

    U1 --> SUM((+))
    U2 --> SUM

    SUM --> OUT["Predicted Noise ε_θ<br>(2 Channels)"]
```

#### 3.1 Dual-Stream Additive Synthesis (Full Context)
The architecture isolates indoor routines from outdoor irrigation conceptually by using two parallel networks, but **crucially, both streams receive the full combined context** (Calendar + Weather).
*   `UNet_in` focuses on baseline diurnal routines.
*   `UNet_out` focuses on massive irrigation spikes.
The final noise prediction is simply $\epsilon_\theta = \epsilon_{in} + \epsilon_{out}$.

#### 3.2 FiLM Layers
Instead of concatenating conditions to the input, the encoded vectors $c$ shift and scale the intermediate activations $h$ of the U-Net's Residual Blocks:

$$
h' = h \odot \gamma(c) + \beta(c)
$$

#### 3.3 U-Net Architecture

<details>
<summary><b>Click to expand: Internal 1D U-Net Tensor Mathematics</b></summary>
<br>

```mermaid
graph TD
    %% Styling
    classDef input fill:#fcf4cd,stroke:#333,stroke-width:1px;
    classDef block fill:#cde9ce,stroke:#333,stroke-width:1px;
    classDef op fill:#ffffff,stroke:#333,stroke-width:1px;
    classDef skip stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;

    subgraph Conditioning ["Conditioning Context"]
        T["Time Embedding<br>(1, 64)"]:::input
        C["Covariates (c_in + c_out)<br>(1, 128)"]:::input
        CatCond(("Concat")):::op
        Context["Combined Context c<br>(1, 192)"]:::block
        
        T --> CatCond
        C --> CatCond
        CatCond --> Context
    end

    subgraph Encoder ["Encoder (Contracting Path)"]
        X["x_t: Noisy Input<br>(1, 2, 96)"]:::input
        
        RB1["ResBlock 1<br>+ FiLM Injection<br>Out: (1, 64, 96)"]:::block
        Pool1["AvgPool1d<br>Out: (1, 64, 48)"]:::op
        
        RB2["ResBlock 2<br>+ FiLM Injection<br>Out: (1, 128, 48)"]:::block
        Pool2["AvgPool1d<br>Out: (1, 128, 24)"]:::op
    end

    subgraph Bottleneck ["Bottleneck"]
        RB3["ResBlock 3<br>+ FiLM Injection<br>Out: (1, 128, 24)"]:::block
    end

    subgraph Decoder ["Decoder (Expansive Path)"]
        Up1["Upsample 1d<br>Out: (1, 128, 48)"]:::op
        Cat1(("Concat")):::op
        RB4["ResBlock 4<br>+ FiLM Injection<br>Out: (1, 64, 48)"]:::block
        
        Up2["Upsample 1d<br>Out: (1, 64, 96)"]:::op
        Cat2(("Concat")):::op
        RB5["ResBlock 5<br>+ FiLM Injection<br>Out: (1, 64, 96)"]:::block
    end

    subgraph Output ["Output Head"]
        FinalConv["Conv1d<br>Out: (1, 2, 96)"]:::block
        Eps["Predicted Noise ε_θ<br>(1, 2, 96)"]:::input
    end

    %% Main Data Flow
    X --> RB1
    RB1 --> Pool1
    Pool1 --> RB2
    RB2 --> Pool2
    Pool2 --> RB3

    RB3 --> Up1
    Up1 --> Cat1
    Cat1 -->|"(1, 256, 48)"| RB4
    
    RB4 --> Up2
    Up2 --> Cat2
    Cat2 -->|"(1, 128, 96)"| RB5
    
    RB5 --> FinalConv
    FinalConv --> Eps

    %% Skip Connections
    RB2 -.->|"Skip Connection<br>(1, 128, 48)"| Cat1
    RB1 -.->|"Skip Connection<br>(1, 64, 96)"| Cat2

    %% Conditioning Flow
    Context -.->|"FiLM parameters (γ, β)"| RB1
    Context -.->|"FiLM parameters (γ, β)"| RB2
    Context -.->|"FiLM parameters (γ, β)"| RB3
    Context -.->|"FiLM parameters (γ, β)"| RB4
    Context -.->|"FiLM parameters (γ, β)"| RB5
```

</details>

### 4. Diffusion Mathematics

The model operates under the framework of Denoising Diffusion Probabilistic Models (DDPM).

#### 4.1 Forward Process (Adding Noise)
We define a fixed variance schedule $\beta_1, \dots, \beta_T$. Let $\alpha_t = 1 - \beta_t$ and:

$$
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

The forward process corrupts the true 2-channel data $x_0$ with Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$:

$$
q(x_t \mid x_0) = \mathcal{N}(x_t ; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I)
$$

Which yields the closed-form sampling step:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon
$$

#### 4.2 Training Objective
The network learns to reverse the process by predicting the injected noise $\epsilon$. Because the model outputs a 2-channel tensor matching the shape of $x_0$, a standard Mean Squared Error (MSE) loss simultaneously trains both the mask and the magnitude objectives:

$$
\mathcal{L}_{MSE} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t, c_{in}, c_{out}) \right\|^2 \right]
$$

#### 4.3 Reverse Process & Inference (Bernoulli Gating)
During sampling, we start with pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoise using the trained model:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right) + \sqrt{\beta_t} z
$$

Once we reach $x_0$, we possess a 2-channel array: $\hat{m}$ (Mask Logits) and $\hat{v}$ (Normalized Log-Magnitude). We pass this through a deterministic **Bernoulli Gate** to perfectly reconstruct the zero-inflated real-world usage:

```mermaid
graph LR
    A["x_0_hat: 2-Channels"] --> B["Channel 0: Mask m"]
    A --> C["Channel 1: Magnitude v"]
    B --> D{"m > 0.5?"}
    C --> E["gallons = exp(v*σ + μ) - 1"]
    D -- Yes --> F["Output = gallons"]
    D -- No --> G["Output = 0.000"]
    E --> F
```

Mathematically, the final generated water usage $y$ is:

$$
y = \mathbb{1}(\hat{m} > 0.5) \cdot \max\left( \exp(\hat{v} \cdot \sigma + \mu) - 1, 0 \right)
$$

---

## Future Directions & Proof of Concept
This repository is primarily a **proof of concept** demonstrating the feasibility of using 1D Conditional Diffusion Models for residential water usage simulation. 

While the current implementation operates on 15-minute intervals (the standard for most Advanced Metering Infrastructure), the architectural framework is designed to be scale-invariant. Ideally, for high-fidelity end-use disaggregation research, this model would be trained on **sub-minute temporal resolution data** (e.g., 1-second or 5-second intervals). At higher resolutions, the diffusion model would be better equipped to capture the fine-grained "signatures" of individual fixtures, such as the distinct flow characteristics of a specific dishwasher cycle or the unique pressure-drop signature of a shower.
