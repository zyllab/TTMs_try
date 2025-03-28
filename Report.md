# Tiny Time Mixers (TTM) Report

## 1. INTRO

In the field of multivariate time series forecasting, traditional models struggle with generalization, high computational cost, and lack of scalability. Recently, large pre-trained models (like GPT4TS, TimesFM) have shown promise, but they require significant resources.

This project explores **Tiny Time Mixers (TTM)** — a lightweight, pre-trained time series forecasting model that achieves strong performance using only ~1M parameters. Firstly, the structure of TTM, TSMixer, PatchTST will be shown for comprehension (TSMixer and PatchTST are two models where some ideas of TTM inherits from). Secondly, this report will record the results of the public demo notebooks from author. Moreover, during the reproduce process, I tried the Mackey-Glass Time Series, for a better understanding of TTM. By doing these, I want to figure out the following questions:

* what is the structure of TTM
* why is TTM constructed in this structure (where does the idea come from)
* what are the advantages of TTM
* where to improve TTM

Ref: [TTM](https://arxiv.org/pdf/2401.03955), [TSMixer](https://arxiv.org/pdf/2306.09364), [PatchTST](https://arxiv.org/abs/2211.14730), [Git Repo of TSFM (Time Series Foundation Models)](https://github.com/ibm-granite/granite-tsfm/blob/main/README.md), [Mackey-Glass Time Series](https://www.kaggle.com/datasets/arashabbasi/mackeyglass-time-series)

---

## 2. Model Components

### PatchTST

<img src="images/report_img/image-20250320032559906.png" alt="image-20250320032559906" style="zoom:40%;" />

* **Work flow of PatchTST**: A multivariate time series sample is first split into separate univariate series  ($x^{(i)}$, one for each variable/channel). Each univariate series is normalized and then divided into fixed-length patches ($x_p^{(i)}$, subsequences). These patches are linearly projected and enriched with positional encoding to retain the order of the time steps. Each channel is processed **independently through a shared Transformer encoder**, where the attention mechanism captures temporal patterns within each series. The encoded patch representations are then flattened and passed through a linear layer to generate the future predictions for each channel. Finally, all channel outputs are combined to form the multivariate forecast.
* **Innovations**: 
  * Channel-Independent Processing
  * Patch-Based Input Representation: reduce the complexity while using **positional embedding to capture the information between each patch**

### TSMixer

<img src="images/report_img/image-20250320035227920.png" alt="image-20250320035227920" style="zoom:27%;" /><img src="images/report_img/image-20250320035458988.png" alt="image-20250320035458988" style="zoom:33.5%;" /><img src="images/report_img/image-20250320035337366.png" alt="image-20250320035337366" style="zoom:30%;" />

* **Work flow of TSMixer**: A multivariate time series sample is first normalized using instance normalization. Then, each univariate channel is divided into patches (small segments), and these patches are reshaped and passed into the TSMixer backbone, which consists of **stacked MLP-Mixer layers**. These layers capture dependencies between patches (temporal), within-patch features, and (optionally) across channels. A **lightweight gated attention** mechanism within each mixer block helps the model focus on important features. After the backbone, a prediction head generates the base forecast. Optionally, two online reconciliation heads refine the forecast: one adjusts it based on cross-channel relationships, and the other ensures consistency across temporal hierarchies (like patch-level aggregations). The final output is a tuned multivariate forecast.
* **Innovations**: Unlike PatchTST, TSMixer use MLP-only layer to capture the information **between patches**, and also **dependence information between channels**. TSMixer achieved these by using **permutations and shared matrix across other fixed dimensions**.

### TTM

<img src="images/report_img/image-20250320041502155.png" alt="image-20250320041502155" style="zoom:50%;" />

* **Work flow of TTM**: A multivariate time series sample is first normalized per channel, then split into non-overlapping patches. Each channel is **treated independently**, and the patches are embedded into a hidden space. A **resolution prefix token** may be prepended to inform the model of the time resolution (e.g., hourly or minutely). The patched sequence is passed through the TTM backbone, a multi-level stack of lightweight TSMixer blocks interleaved with **gated attention**, where patch dimensions are **adaptively reshaped at each level** to learn both local and global patterns. During fine-tuning, the backbone is frozen, and a slim **decoder with optional channel mixing** produces the initial forecast. If exogenous variables are available, they are fused using an **exogenous mixer**, which patches and mixes the known future values with forecasted targets to refine the prediction. Finally, the output is passed through a linear head and reverse-normalized to produce the final multivariate forecast.
* **Innovations**
  * Using **Adaptive Patching (AP)** and **Resolution Prefix Tuning (RPT)**, TTM generalizes well on datasets with different resolutions. It capture the information of different length pattern (resolution)  by using multiple layers of TSMixer block specifying different input lengths.
  * Only considering the channel-independent situation in pre-trained model, **leaving the fine-tune part (Decoder) to capture the dependence between channels** enlarges the datasets to pre-train the model, enhanced the performance.
  * Using **Exogenous Mixer** to learn lagged dependencies between known future exogenous inputs and target variables for more accurate forecasting.

---

## 3. Notebooks Results

### A. [Get started withe TinyTimeMixer(TTM)](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb)

This notebook shows how an example to use TTM (changing loss function, context/forecast length, zero/few-shot) on data [ETTH1](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv). Also, I perform the same task on simulated Mackey-Glass Time Series with 11000 time steps. Here are the results. For simplicity, I use 512-96 to represent context length 512 and forecast length 96

* ETTH1, Zero-shot, 512-96

<img src="images/report_img/e-0-96.png" alt="e-0-96" style="zoom: 67%;" />

* ETTH1, few-shot, 512-96, MSE loss

<img src="images/report_img/e-5-96-m.png" alt="e-5-96-m" style="zoom:67%;" />

* EHHT1, few-shot, 512-96, quantile loss

<img src="images/report_img/e-5-96-q.png" alt="e-5-96-q" style="zoom:67%;" />

* EHHT1, zero-shot, 1024-48

<img src="images/report_img/e-0-48.png" alt="e-0-48" style="zoom:67%;" />

* EHHT1, few-shot, 1024-48

<img src="images/report_img/e-5-48.png" alt="e-5-48" style="zoom:67%;" />

* MG, zero-shot, 512-96

<img src="images/report_img/m-0-96.png" alt="m-0-96" style="zoom:67%;" />

* MG, few-shot, 512-96, MSE loss

<img src="images/report_img/m-5-96-m.png" alt="m-5-96-m" style="zoom:67%;" />

* MG, few-shot, 512-96, quantile loss

<img src="images/report_img/m-5-96-q.png" alt="m-5-96-q" style="zoom:67%;" />

* MG, zero-shot, 1024-48

<img src="images/report_img/m-0-48.png" alt="m-0-48" style="zoom:67%;" />

* MG, few-shot, 1024-48

<img src="images/report_img/m-5-48.png" alt="m-5-48" style="zoom:67%;" />

Here is the test MSE:

* EHHT1

|  | 512-96 MSE  |512-96 quantile |1024-48|
| ---- | ---------- | --------------- | ------- |
| zero-shot | 0.363 |/|0.333|
| few-shot| 0.362  |0.362|0.333|

* Mackey-Glass

|      | 512-96 MSE | 512-96 quantile | 1024-48 |
| ---- | ---------- | --------------- | ------- |
| zero-shot | 1.122 |/|1.010|
| few-shot|    0.693  |0.726|0.682|

* **Insights**: We can see from the table that:
  * Over ETTH1, the loss after and before fine-tuning does not differ a lot, indicating good performance of the pre-trained model
  * Over Mackey-Glass time series data, the loss decreased a lot after fine tuning, indicating the transfer part can make the model well fit to some dataset that is far from training dataset. 
  * Although fine tuning decreased the loss a lot, but is is far from satisfaction, indicating the lack of ability to capture chaotic dynamics
  * Lengthen the context length and shorten the forecast length can decrease the loss significantly on ETTH1 (real world data), while the loss on MG remains almost unchanged. Which is reasonable since the information of previous time step can last at most $\tau$ steps according to its formula, $\frac{dx(t)}{dt} = \beta \frac{x(t - \tau)}{1 + x(t - \tau)^n} - \gamma x(t)$

### B. [Getting started with TinyTimeMixer (TTM) Rolling Predictions](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/ttm_rolling_prediction_getting_started.ipynb)

This notebook showed the rolling prediction by the TTM, context length 512 and forecast length 96, on the data ETTH1.  For the rolling prediction length 192, the zero shot MSE is 0.392. Here is the prediction plot

<img src="images/report_img/rolling-pred.png" alt="rolling-pred" style="zoom:67%;" />

* **Insight**: We can see that though we use the rolling prediction, but the loss did not increase largely, indicating that this model performs very well on prediction of ETTH1.

### C. [Getting started with PatchTSMixer](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/patch_tsmixer_getting_started.ipynb)

This notebook showed the how to use the pre-trained model (trained in ETTH1) on ETTH1, also showed how to change hyper-parameter and change the target channel during training from scratch. Both the pre-trained model and model training from scratch got evaluation loss around 0.368.

* Pre-trained:

<img src="images/report_img/image-20250320085910842.png" alt="image-20250320085910842" style="zoom:80%;" />

* Train from scratch:

<img src="images/report_img/image-20250320085956401.png" alt="image-20250320085956401" style="zoom:80%;" />

* **Insights**: We can see the evaluation loss is not far from TTM, which is reasonable since the most advantage of TTM is that it is pre-trained and can generalize well.

### D. [Patch Time Series Mixer for Transfer Learning across datasets](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/patch_tsmixer_transfer.ipynb) 

This notebook showed how to do transfer learning of TSMixer. Firstly, this model is pre-trained on the [ECL](https://github.com/zhouhaoyi/Informer2020) data, and then use the saved model to predict on the ETTH2 dataset. It used the zero-shot, linear probing and the full fine-tune:

* Zero-shot

<img src="images/report_img/image-20250320090758096.png" alt="image-20250320090758096" style="zoom:80%;" />

* Linear probing

![image-20250320090834240](images/report_img/image-20250320090834240.png)

* Full fine-tune:

![image-20250320090913570](images/report_img/image-20250320090913570.png)

* **Insights**: evaluation loss decreases lot on the linear probing but increase when doing full fine-tuning. Indicating that for ETTH2, linear probing is enough.

### E. [PatchTSMixer in HuggingFace - Getting Started](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/patchtsmixer_HF_blog.ipynb)

The same with D. [Patch Time Series Mixer for Transfer Learning across datasets](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/patch_tsmixer_transfer.ipynb)

 ### F. [PatchTSMixer in HuggingFace - Getting Started](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/patch_tsmixer_blog.ipynb)

The same with D. [Patch Time Series Mixer for Transfer Learning across datasets](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/patch_tsmixer_transfer.ipynb)

### G. [Getting started with PatchTST](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/patch_tst_getting_started.ipynb)

This notebook showed the how to use the pre-trained model (trained in ETTH1) on ETTH1, also showed how to change hyper-parameter and change the target channel during training from scratch. 

* Pre-trained:

![image-20250320092315845](images/report_img/image-20250320092315845.png)

* Train from scratch:

![image-20250320092340505](images/report_img/image-20250320092340505.png)

### H. [Patch Time Series Transformer for Transfer Learning across datasets]()

This notebook showed how to do transfer learning of PatchTST. Firstly, this model is pre-trained on the [ECL](https://github.com/zhouhaoyi/Informer2020) data, and then use the saved model to predict on the ETTH2 dataset. It used the zero-shot, linear probing and the full fine-tune:

* Zero-shot:

![image-20250320095550001](images/report_img/image-20250320095550001.png)

* Linear probing:

![image-20250320095644498](images/report_img/image-20250320095644498.png)

* Full fine tune:

![image-20250320095719913](images/report_img/image-20250320095719913.png)

* **Insights**: evaluation loss decreases lot on the linear probing but increase when doing full fine-tuning. Indicating that for ETTH2, linear probing is enough.


---

## 4. Experiments

### A. Few-shot percentage vs Time delay

This experiment mainly consider the performance of the model with different percentage in the few-shot training over different MG data with different time delay $\tau$.

Firstly, in the TTMs, the percentage in few-shot means the fraction of samples for fine-tuning (slicing windows with $length = context\_length+forecast\_length$). There are three ways to select the training samples, from start, last and randomly choose from training dataset. We will do both the last position and the uniformly choosing.

For simulation of the MG data, I take the [tutorial](https://www.mathworks.com/matlabcentral/fileexchange/24390-mackey-glass-time-series-generator) for reference, using the Runge-Kutta method to numerically approaching the derivative. Here is a scratch:

To generate

$$
\frac{dP(t)}{dt} = \frac{\beta\theta^nP(t-\tau)}{\theta^n+P(t-\tau)^n}-\gamma P(t)
$$

We use the Runge-Kutta method, with $\frac{dy}{dt} = f(t,y)$ known:

$$
\begin{aligned}
y_{n+1} &= y_n + \frac{h}{6}(k_1+2k_2+2k_3+k_4)\\
t_{n+1} &= t_n+h\\
k_1 & = f(t_n,y_n)\\
k_2& = f(t_n+\frac{h}{2},y_n+h\frac{k_1}{2})\\
k_3 &= f(t_n+\frac{h}{2},y_n+h\frac{k_2}{2})\\
k_4 & = f(t_n+h,y_n+hk_3)
\end{aligned}
$$

We chose the parameters to be $\theta = 0.2,\beta = 0.2,\gamma = 0.1,n = 10,\tau\in\{20,30,40,100\}$. 

For different performance of different parameters, see the streamlit scratch.

Here the table of MSEs

* Select from last position

|$\tau$|Zero-shot|5%|20%|30%|40%|50%|75%|
|---|---|---|---|---|---|---|---|
|20|1.120|0.870|0.734|0.692|0.334|0.192|0.103|
|30|1.198|0.923|0.678|0.480|0.403|0.379|0.221|
|40|0.245|0.212|0.251|0.113|0.067|0.047|0.047|
|100|1.186|0.989|0.879|0.371|0.234|0.181|0.137|

* Select randomly

|$\tau$|Zero-shot|5%|20%|30%|40%|50%|75%|
|---|---|---|---|---|---|---|---|
|20|1.120|0.492|0.221|0.195|0.136|0.117||
|30|1.198|0.563|0.298|0.250|0.221|0.211||
|40|0.245|0.199|0.109|0.087|0.067|0.063||
|100|1.186|0.467|0.267|0.225|0.191|||



---

## 5. Summary

This report explores and evaluates the **Tiny Time Mixers (TTM)** model by comparing it with related models (**PatchTST** and **TSMixer**) and running experiments on real-world and synthetic time series datasets. The study highlights the following key takeaways:

- **Architecture-wise**, TTM adopts and enhances ideas from PatchTST and TSMixer:
  - It uses **adaptive patching** and **resolution prefix tuning** to generalize across datasets with different temporal resolutions.
  - By **freezing the backbone** during fine-tuning and focusing channel dependencies in the decoder, TTM supports large-scale pretraining without losing flexibility.
  - The **Exogenous Mixer** module makes it capable of incorporating known future variables for improved forecasting.
- **Experimental results** show:
  - TTM performs **strongly out-of-the-box** (zero-shot) on standard datasets like ETTH1.
  - It also **adapts well through fine-tuning** to datasets with different dynamics, such as the Mackey-Glass time series.
  - However, its performance on chaotic time series (like Mackey-Glass) remains limited, suggesting room for improvement in handling non-linear dynamics.
- **Model transfer experiments** with TSMixer and PatchTST reveal that:
  - **Linear probing** is often sufficient for transfer learning and outperforms full fine-tuning in some cases.
  - Pre-trained models can match or exceed models trained from scratch, especially when data distributions are similar.

Overall, **TTM is a powerful and lightweight solution** for time series forecasting that balances pretraining efficiency with flexible adaptation. Future work can focus on better capturing chaotic behavior and further optimizing fine-tuning strategies.
