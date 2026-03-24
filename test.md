---
marp: true
theme: default
paginate: true
math: katex
style: |
  section { font-size: 24px; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .box { padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
  h1 { color: #2c3e50; }
  h2 { color: #34495e; }
  strong { color: #e74c3c; }
---

# Simple and Effective Masked Diffusion Language Models

<div style="text-align: center;">

> [**Paper Link:** https://arxiv.org/abs/2406.07524](https://arxiv.org/pdf/2406.07524)
</div>

<div style="text-align: center;"> 

**Presenters:** Qihang Zhang, Donglin Yang

</div>
<br>
<div style="text-align: center;">2025, Dec 3rd</div>

---

## Overview: The Gap in Discrete Diffusion

*   **Context:** Diffusion models dominate image generation but lag behind Autoregressive (AR) models in text (Log-Likelihood gap).
*   **The Solution:** MDLM (Masked Diffusion Language Model).
*   **Key Insight:** Simplify discrete diffusion to a **weighted average of Masked Language Modeling (MLM) losses**.
*   **Result:** New State-of-the-Art (SOTA) among diffusion models, approaching AR perplexity.

---
<img src="assets/mdlm.png" alt="MDLM Diagram" style="width:100%; height:auto;">
---

## **Continuous vs. Discrete Diffusion Models**

| Feature | Continuous Diffusion (DDPM) | Masked Discrete Diffusion (MDLM) |
| :--- | :--- | :--- |
| **Data Space** | <div style="background-color: #e8f4ff; padding: 10px; border-radius: 5px;">Continuous $x \in \mathbb{R}^d$</div> | <div style="background-color: #ffeaea; padding: 10px; border-radius: 5px;">Discrete tokens $x \in \mathcal{V}$ (one-hot)</div> |
| **Noise Source** | <div style="background-color: #e8f4ff; padding: 10px; border-radius: 5px;">Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$</div> | <div style="background-color: #ffeaea; padding: 10px; border-radius: 5px;">Masking state $m$ (`[MASK]`)</div> |
| **Forward Process** | <div style="background-color: #e8f4ff; padding: 10px; border-radius: 5px;">Gradually add Gaussian noise until signal is destroyed.</div> | <div style="background-color: #ffeaea; padding: 10px; border-radius: 5px;">Gradually replace tokens with `[MASK]` (absorbing state).</div> |
| **Reverse Process** | <div style="background-color: #e8f4ff; padding: 10px; border-radius: 5px;">Predict the **noise** to denoise steps.</div> | <div style="background-color: #ffeaea; padding: 10px; border-radius: 5px;">Predict **original tokens** $x_0$ to "unmask".</div> |


---

### 1. The Forward Process: Comparison

<div style="width: 100%; overflow-x: auto;">
  <table style="width: 100%; font-size: 0.55em; border-collapse: collapse;">
    <thead>
      <tr style="border-bottom: 2px solid #fff;">
        <th style="text-align: left; padding: 10px; width: 15%;">Feature</th>
        <th style="text-align: left; padding: 10px; width: 42%;">Continuous (Standard DDPM)</th>
        <th style="text-align: left; padding: 10px; width: 43%;">MDLM (Masked Diffusion)</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #555;">
        <td style="padding: 8px;"><strong>Core Mechanism</strong></td>
        <td style="padding: 8px;">Additive Gaussian Noise</td>
        <td style="padding: 8px;">Interpolation / Masking</td>
      </tr>
      <tr style="border-bottom: 1px solid #555;">
        <td style="padding: 8px;"><strong>Transition Function</strong></td>
        <td style="padding: 8px;">$$q(\mathbf{z}_t\|\mathbf{z}_{t-1}) = \mathcal{N}(\mathbf{z}_t; \sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t \mathbf{I})$$</td>
        <td style="padding: 8px;">
            $$q(\mathbf{z}_t\|\mathbf{z}_s) = \text{Cat}(\mathbf{z}_t; \mathbf{Q}_{t\mid s}^\top z_s) =  \text{Cat}(\mathbf{z}_t; \alpha_{t\mid s} \mathbf{z}_{s} + (1-\alpha_{t\mid s})\mathbf{m})$$
            <br>
            $$Q_{t\mid s}=\alpha_{t\mid s}\mathbf{I}+ (1-\alpha_{t\mid s})\mathbf{1}\mathbf{m}^\top, \alpha_{t\mid s}=\frac{\alpha_t}{\alpha_s}$$
        </td>
      </tr>
      <tr style="border-bottom: 1px solid #555;">
        <td style="padding: 8px;"><strong>Marginal Distribution</strong> $q(\mathbf{z}_t\|\mathbf{x})$</td>
        <td style="padding: 8px;">$$\mathcal{N}(\mathbf{z}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}, (1-\bar{\alpha}_t)\mathbf{I})$$</td>
        <td style="padding: 8px;">$$\text{Cat}(\mathbf{z}_t; \alpha_t \mathbf{x} + (1-\alpha_t)\mathbf{m})$$</td>
      </tr>
      <tr style="border-bottom: 1px solid #555;">
        <td style="padding: 8px;"><strong>State Space</strong></td>
        <td style="padding: 8px;">
            <strong>Continuous</strong> ($\mathbf{z}_t \in \mathbb{R}^D$)<br>Value shifts gradually; info is obscured by noise variance.
        </td>
        <td style="padding: 8px;">
            <strong>Discrete (Absorbing)</strong> ($\mathbf{z}_t \in \{\mathbf{x}, \mathbf{m}\}$)<br>Token is either clean or strictly <code>[MASK]</code>. Once masked, info is lost (absorbing).
        </td>
      </tr>
      <tr>
        <td style="padding: 8px;"><strong>Schedule</strong></td>
        <td style="padding: 8px;">Variance schedule $\beta_t$ (or $\alpha_t$) controls noise level.</td>
        <td style="padding: 8px;">$\alpha_t$ is strictly decreasing ($1 \to 0$); represents probability of token being unmasked.</td>
      </tr>
    </tbody>
  </table>
</div>

---

## 2. The Forward Posterior: Derivation

We need $q(\mathbf{z}\_s | \mathbf{z}\_t, \mathbf{x})$ where $s < t$ (less noisy).
From D3PM (Austin et al.), the posterior for discrete diffusion is given by Bayes rule involving the transition matrices $Q$:

$$
q(\mathbf{z}_s|\mathbf{z}_t, \mathbf{x}) = \text{Cat}\left\( \mathbf{z}_s; \frac{Q _{t|s}\mathbf{z} _t \odot Q _s^\top \mathbf{x}}{\mathbf{z} _t^\top Q _t^\top \mathbf{x}} \right) 
$$

Where:
*   $Q_{t|s} = \alpha_{t|s}\mathbf{I} + (1-\alpha_{t|s})\mathbf{1}\boldsymbol{\pi}^\top$ (Transition from $s \to t$)
*   $\odot$ is the element-wise (Hadamard) product.
*   $\boldsymbol{\pi}$ is the stationary distribution (the noise prior).

---
We substitute the definition of $Q$ into the posterior,

$$
q(\mathbf{z}_s|\mathbf{z}_t, \mathbf{x}) = \text{Cat}\left\(\mathbf{z}_s; \frac{[\alpha\_{t|s}\mathbf{I} + (1-\alpha\_{t|s})\mathbf{1}\boldsymbol{\pi}^\top]\mathbf{z} _t \odot [\alpha\_s\mathbf{I} + (1-\alpha\_s)\mathbf{1}\boldsymbol{\pi}^\top]^\top \mathbf{x}}{\mathbf{z}_t^\top[\alpha\_t\mathbf{I} + (1-\alpha\_t)\mathbf{1}\boldsymbol{\pi}^\top]^\top \mathbf{x}}\right\)
$$
$$
= \text{Cat}\left\(\mathbf{z}_s; \frac{[\alpha _{t|s}\mathbf{z} _t + (1-\alpha _{t|s})\mathbf{1}(\boldsymbol{\pi}^\top \mathbf{z} _t)] \odot [\alpha _s\mathbf{x} + (1-\alpha _s)\boldsymbol{\pi}]}{\alpha\_t \mathbf{z}_t^\top\mathbf{x}+(1-\alpha_t)\mathbf{z}_t^\top\boldsymbol{\pi}}\right\)
$$

Now, we apply this to **Masked Diffusion**.
*   The prior $\boldsymbol{\pi} = \mathbf{m}$ (the Mask token).
*   State space is discrete: $\mathbf{z}_t \in \{\mathbf{x}, \mathbf{m}\}$.

We analyze two cases: **Unmasked** and **Masked**.

---

## Case 1: The Unmasked Token ($\mathbf{z}_t = \mathbf{x}$)

If the token at time $t$ is **not** masked ($\mathbf{z}_t = \mathbf{x}$), it must have been unmasked at time $s$ (since masking is absorbing).
Substitute $\mathbf{z}\_t = \mathbf{x}$ and $\boldsymbol{\pi} = \mathbf{m}$, and
note that $\mathbf{x} \odot \mathbf{m} = 0$ (orthogonal vectors) and $\mathbf{x}^\top \mathbf{m} = 0$.

$$
q(\mathbf{z}\_s|\mathbf{z}\_t=\mathbf{x}, \mathbf{x}) = \text{Cat}\left\(\mathbf{z}\_s;\frac{[\alpha_{t|s}\mathbf{x} + (1-\alpha_{t|s})\mathbf{1}\overbrace{\mathbf{m}^\top \mathbf{x}}^{0}] \odot [\alpha_s\mathbf{x} + (1-\alpha_s)\mathbf{m}]}{\alpha_t \mathbf{x}^\top \mathbf{x}+(1-\alpha_t)\mathbf{x}^\top\mathrm{m}}\right\)
$$

$$
= \text{Cat}\left\(\mathbf{z}\_s;\frac{[\alpha_{t|s}\mathbf{x}] \odot [\alpha_s\mathbf{x} + (1-\alpha_s)\mathbf{m}]}{\alpha_t}\right\)=\text{Cat}\left\(\mathbf{z}\_s;\mathbf{x}\right\)=\text{Cat}\left\(\mathbf{z}\_s;\mathbf{z}\_t\right\)
$$
---

## Case 2: The Masked Token ($\mathbf{z}_t = \mathbf{m}$)

If the token is currently masked, it was either **already masked** at time $s$, or it **became masked** between $s$ and $t$. Substitute $\mathbf{z}_t = \mathbf{m}$ and $\boldsymbol{\pi} = \mathbf{m}$, and note that $\mathbf{m}^\top \mathbf{m} = 1$ and $\mathbf{m} \odot \mathbf{m} = \mathbf{m}$.

$$
q(\mathbf{z}\_s|\mathbf{z}\_t=\mathbf{x}, \mathbf{x}) = \text{Cat}\left\(\mathbf{z}\_s;\frac{[\alpha_{t|s}\mathbf{m} + (1-\alpha_{t|s})\mathbf{1}] \odot [\alpha_s\mathbf{x} + (1-\alpha_s)\mathbf{m}]}{(1-\alpha_t)}\right\)
$$

$$
= \text{Cat}\left\(\mathbf{z}\_s;\frac{(1-\alpha_s)\mathbf{m}+(\alpha_s-\alpha_t)\mathbf{x}}{1-\alpha_t}\right\)
$$

---

## Summary: The Posterior $q(\mathbf{z}\_s | \mathbf{z}\_t, \mathbf{x})$

Combining both cases, we arrive at the closed-form posterior used for training:

$$
q(\mathbf{z}\_s|\mathbf{z}\_t, \mathbf{x}) = 
\text{Cat}(\mathbf{z}\_s; \mathbf{z}\_t) \quad \text{if } \quad \mathbf{z}\_t \neq \mathbf{m}
$$

$$
q(\mathbf{z}\_s|\mathbf{z}\_t, \mathbf{x}) = 
\text{Cat}\left\(\mathbf{z}\_s;\frac{(1-\alpha_s)\mathbf{m}+(\alpha_s-\alpha_t)\mathbf{x}}{1-\alpha_t}\right\) \quad \text{if} \quad \mathbf{z}\_t = \mathbf{m} 
$$


**Key Insight:** If the token is masked, the posterior is weighted by the noise schedule difference. If the token is unmasked, it will not be changed.


---

## 3. The Reverse Process: Parameterization

How do we model $p_\theta(\mathbf{z}\_s | \mathbf{z}\_t)$ to approximate the posterior?

**SUBS Parameterization** 
Network $x_\theta(\mathbf{z}\_t, t)$ predicts clean $x$.

$$
p_\theta(\mathbf{z}\_s|\mathbf{z}\_t) = 
\text{Cat}(\mathbf{z}\_s; \mathbf{z}\_t) \quad \text{if } \quad \mathbf{z}\_t \neq \mathbf{m}
$$

$$
p_\theta(\mathbf{z}\_s|\mathbf{z}\_t) = 
\text{Cat}\left\(\mathbf{z}\_s;\frac{(1-\alpha_s)\mathbf{m}+(\alpha_s-\alpha_t)\mathbf{x_\theta(\mathbf{z}_t)}}{1-\alpha_t}\right\) \quad \text{if} \quad \mathbf{z}\_t = \mathbf{m} 
$$

**Logic:** If token is visible, keep it. If masked, use network prediction.
---

## Deep Dive: The SUBS Parameterization

The authors introduce **SUBS** (Substitution) to enforce logic:

1.  **Zero Masking Probabilities:**
    $\langle x_\theta(z_t, t), m \rangle = 0$.
    *   The network never predicts `[MASK]` as the original token.
    *   *Implementation:* Set logit for `[MASK]` to $-\infty$.

2.  **Carry-Over Unmasking:**
    If $z_t$ is unmasked, $x_\theta$ is ignored/overwritten by $z_t$.
    *   *Result:* Unmasked tokens remain unchanged during reverse diffusion.

---

## 4. The Objective Function (Rao-Blackwellized Likelihood Bounds)

The discrete-time diffusion loss for finite $T$.

$$
\mathcal{L}\_{\text{diffusion}} = \sum_{i=1}^{T} \mathbb{E}\_{q} \left[ \text{D}\_{\text{KL}} \left( q(\mathbf{z}\_{s(i)} | \mathbf{z}\_{t(i)}, \mathbf{x}) \middle\| p\_{\theta}(\mathbf{z}\_{s(i)} | \mathbf{z}\_{t(i)}) \right) \right]
$$
$$
= \sum\_{i=1}^{T} \mathbb{E}\_{q} \left[ \frac{\alpha_{t(i)} - \alpha_{s(i)}}{1 - \alpha_{t(i)}} \log \langle \mathbf{x}\_{\theta}(\mathbf{z}\_{t(i)}), \mathbf{x} \rangle \right]
$$

*   **Weighted Cross-Entropy.**
*   Only computed on masked tokens.

---

## Deriving the Objective

The discrete diffusion loss is $\mathbb{E}_q [ D\_{KL}(q(z_s|z_t, x) || p_\theta(z_s|z_t)) ]$.

1.  **Case 1: $z_t = x$ (Unmasked)**
    *   $q(z_s|z_t, x)$ is deterministic (stay unmasked).
    *   $p_\theta$ copies $z_t$ (Carry-Over).
    *   $D_{KL} = 0$. (Loss is zero).

2.  **Case 2: $z_t = m$ (Masked)**
    *   Simplifies to a standard Cross Entropy term scaled by diffusion schedule:
    $$ \text{Loss} \propto - \log p_\theta(x | z_t \text{ is masked}) $$

---

## 5. Continuous Time Extension ($T \to \infty$)

The authors extend this to continuous time for better performance (Eq 10).

$$ \mathcal{L}\_{\text{NELBO}}^\infty = \mathbb{E}\_{q} \int_{0}^{1} \frac{\alpha'\_t}{1-\alpha_t} \log \langle x_\theta(z_t, t), x \rangle dt $$

*   **Interpretation:**
    *   $\frac{\alpha'_t}{1-\alpha_t}$: Weighting function based on noise schedule.
    *   $\log \langle \dots \rangle$: Standard Cross Entropy .
    
---
## 6. Masked Diffusion Language Models

Next, the authors apply masked diffusion to language modeling over sequences $\mathbf{x}^{1:L}$ of $L$ tokens. The forward noising process is applied independently accross a sequence, and $p_\theta({\mathbf{z}_s^{1:L}\mid \mathbf{z}_t^{1:L}})=\Pi\_{l=1}^{L} p_\theta(\mathbf{z}^l_s\mid  \mathbf{z}_t^{1:L})$.

$$ \mathcal{L}\_{\text{NELBO}}^\infty = \mathbb{E}\_{q} \int_{0}^{1} \frac{\alpha'\_t}{1-\alpha_t} \sum_{\ell=1}^{L} \log \langle x_\theta^\ell(z_t^{1:L}, t), x^\ell \rangle dt $$

* **Note**: Although the loss imposes a loss on all tokens, **unmasked tokens don’t contribute to the loss**, as they are copied over due to “carry-over unmasking”.
    
**Conclusion:** MDLM training is **weighted Masked Language Modeling**. It establishes a connection between diffusion models and **encoder-only BERT models**.

---

## 7.  Summary of MDLM Method

1.  **Forward:** Independent tokens flip to `[MASK]` over time.
2.  **Reverse:** Predict $x_0$ directly using a Transformer (DiT/BERT).
3.  **Parameterization (SUBS):**
    *   Never predict mask.
    *   Trust unmasked tokens (don't re-predict them).
4.  **Training:**
    *   Sample time $t$.
    *   Mask input $x$ based on $\alpha_t$.
    *   Minimize Cross Entropy on masked tokens weighted by $\frac{\alpha'_t}{1-\alpha_t}$.

---

## MDLM vs Standard MLM

*   **Standard BERT (MLM):**
    *   Mask 15% of tokens once.
    *   Predict at masked positions.
    
*   **MDLM:**
    *   Masking rate varies from 0% to 100% (controlled by $t$).
    *   Loss is weighted.
    *   **Generation:** Can generate text from pure noise (all masks) by iteratively unmasking.

---

## Sampling / Inference Process

<div style="text-align: center;">
  <img src="figures/sampling_v2.png" alt="Sampling Process" style="max-width: 100%; height: auto; display: inline-block;">
</div>

---

## Experiments
+ Language Modelling Benchmarks
+ Downstream NLP Tasks
+ DNA Sequence Modelling
+ Downstream DNA Tasks

---

## Experimental Setting: 

**PPL**: Perplexity evaluates how well the model predicts the reference tokens:

$$
\mathrm{PPL} = \exp \left(\mathbb{E} _{x \sim \mathbb{P} _{data}} - \frac{1}{N(x)} \sum _{i=1}^{N(x)} \log P(x _i \mid x _{<i})\right),
$$

Where:
+ $N(x)$ is the number of tokens of the sequence $x$.
+ PPL is indeed the exponential of the average negative log-likelihood per token.

---
## Calculate PPL for AR and Diffusion Models

- **AR Models**: We can directly compute PPL.
- **Diffusion Models**: 
  - We can only approximate PPL with ELBO.
  - PPL $\leq \exp(-\text{ELBO})$

---

## Language Modeling Results: `LM1B`(in dist.)

<div style="display: flex; gap: 28px; align-items: flex-start; font-size: 24px; line-height: 1.35; max-width: 1180px; margin: 0 auto;">
  <div style="flex: 0 0 62%;">
    <div style="font-size: 28px; margin: 0 0 12px 0;">
      Table 1: Test perplexities (PPL; &darr;) on LM1B. &dagger;Reported in <a href="https://arxiv.org/abs/2211.15029">Diffusion-bert</a>. Best diffusion value is bolded.
    </div>
    <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 24px;">
      <thead>
        <tr>
          <th style="width: 26%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
          <th style="width: 44%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
          <th style="width: 15%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Parameters</th>
          <th style="width: 15%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PPL (&darr;)</th>
        </tr>
      </thead>
      <tbody>
        <tr style="color: #0b63ce;">
          <td style="font-style: italic;" rowspan="2">Autoregressive</td>
          <td>Transformer-X Base [13]</td>
          <td style="text-align: center;">0.46B</td>
          <td style="text-align: center;">23.5</td>
        </tr>
        <tr style="color: #0b63ce;">
          <td style="border-bottom: 1px solid #000;">OmniNet<sub>T</sub> [61]</td>
          <td style="text-align: center; border-bottom: 1px solid #000;">100M</td>
          <td style="text-align: center; border-bottom: 1px solid #000;">21.5</td>
        </tr>
        <tr style="color: #0b63ce;">
          <td style="font-style: italic; border-top: 1px solid #000;" rowspan="5">Diffusion</td>
          <td style="border-top: 1px solid #000;">BERT-Mouth [64]&dagger;</td>
          <td style="text-align: center; border-top: 1px solid #000;">110M</td>
          <td style="text-align: center; border-top: 1px solid #000;">&le;142.89</td>
        </tr>
        <tr style="color: #0b63ce;">
          <td>D3PM (absorb) [1]</td>
          <td style="text-align: center;">70M</td>
          <td style="text-align: center;">&le;76.90</td>
        </tr>
        <tr style="color: #0b63ce;">
          <td>Diffusion-LM [30]&dagger;</td>
          <td style="text-align: center;">80M</td>
          <td style="text-align: center;">&le;118.62</td>
        </tr>
        <tr style="color: #0b63ce;">
          <td>DiffusionBert [26]</td>
          <td style="text-align: center;">110M</td>
          <td style="text-align: center;">&le;63.78</td>
        </tr>
        <tr style="color: #0b63ce;">
          <td style="border-bottom: 1px solid #000;">SEDD [33] (33B tokens)</td>
          <td style="text-align: center; border-bottom: 1px solid #000;">110M</td>
          <td style="text-align: center; border-bottom: 1px solid #000;">&le; 32.79</td>
        </tr>
        <tr style="color: #d2202f;">
          <td style="font-style: italic; border-top: 1px solid #000;" rowspan="2">Autoregressive<br>(Retrained)</td>
          <td style="border-top: 1px solid #000;">Transformer (33B tokens)</td>
          <td style="text-align: center; border-top: 1px solid #000;">110M</td>
          <td style="text-align: center; border-top: 1px solid #000;">22.32</td>
        </tr>
        <tr style="color: #d2202f;">
          <td style="border-bottom: 1px solid #000;">Transformer (327B tokens)</td>
          <td style="text-align: center; border-bottom: 1px solid #000;">110M</td>
          <td style="text-align: center; border-bottom: 1px solid #000;">20.86</td>
        </tr>
        <tr style="color: #d2202f;">
          <td style="font-style: italic; border-top: 1px solid #000;" rowspan="2">Diffusion<br>(Ours)</td>
          <td style="border-top: 1px solid #000;">MDLM (33B tokens)</td>
          <td style="text-align: center; border-top: 1px solid #000;">110M</td>
          <td style="text-align: center; border-top: 1px solid #000;">&le;27.04</td>
        </tr>
        <tr style="color: #d2202f;">
          <td style="border-bottom: 1px solid #000;">MDLM (327B tokens)</td>
          <td style="text-align: center; border-bottom: 1px solid #000;">110M</td>
          <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">&le;23.00</td>
        </tr>
      </tbody>
    </table>
  </div>
  <div style="flex: 1;">
    <div style="font-size: 26px; margin: 0 0 12px 0;">Key takeaways</div>
    <ul style="margin: 0; padding-left: 18px; font-size: 24px; line-height: 1.45;">
      <li><strong>LM1B:</strong> <br> One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling <br><br><strong>Origin:</strong> WMT 2011 English News Crawl<br><strong>Feature:</strong> Only short sentenses, and sentenses are shuffled.</li>
      <li><strong>Context Window:</strong> 128 Tokens</li>
    </ul>
  </div>
</div>

**Takehome Message:** MDLM beats SEDD (previous SOTA). 


---

## Language Modeling Results: `OWT` (in dist.)

<div style="display: flex; gap: 24px; align-items: flex-start; font-size: 26px; line-height: 1.35; max-width: 1180px; margin: 0 auto;">
  <div style="flex: 0 0 62%; max-width: 620px;">
    <div style="margin: 0 0 12px 0;">
      Table 2: Test perplexities (PPL; &darr;) on OWT for models trained for 262B tokens. &dagger; denotes retrained models.
    </div>
    <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 48px;">
      <thead>
        <tr>
          <th style="width: 55%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
          <th style="width: 45%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PPL (&darr;)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="border-bottom: 1px solid #000;">AR&dagger;</td>
          <td style="text-align: center; border-bottom: 1px solid #000;">17.54</td>
        </tr>
        <tr>
          <td style="border-bottom: 1px solid #000;">SEDD&dagger;</td>
          <td style="text-align: center; border-bottom: 1px solid #000;">&le;24.10</td>
        </tr>
        <tr>
          <td style="border-bottom: 2px solid #000;">MDLM (Ours)</td>
          <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;"><strong>&le;23.21</strong></td>
        </tr>
      </tbody>
    </table>
  </div>
  <div style="flex: 1; font-size: 36px; line-height: 1.45;">
    <div style="font-size: 36px; margin: 0 0 10px 0;">OWT Dataset:</div>
    <ul style="margin: 0; padding-left: 18px;">
      <li><strong>Full name:</strong> OpenWebText Corpus</li>
      <li><strong>Description:</strong> Internal dataset used to train GPT-2.</li>
      <li><strong>Context Window:</strong> 1024 Tokens</li>
    </ul>
  </div>
</div>

**Takehome Message:** MDLM outperforms SEDD on OWT dataset as well.

---

## Language Modeling Results (out of dist.)

<div style="font-size: 26px; line-height: 1.35; max-width: 1080px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 3: Zero-shot perplexities (PPL; &darr;) of models trained for 524B tokens on OWT. All perplexities for diffusion models are upper bounds.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 28px;">
    <thead>
      <tr>
        <th style="width: 16%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PTB</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Wikitext</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">LM1B</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Lambada</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">AG News</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PubMed</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Arxiv</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="font-style: italic; border-bottom: 1px solid #000;">AR (Retrained)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">82.05</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">25.75</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">51.25</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">51.28</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">52.09</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">49.01</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">41.73</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-bottom: 1px solid #000;">SEDD (Retrained)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">100.09</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">34.28</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">68.20</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">49.86</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">62.09</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">44.53</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">38.48</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-bottom: 2px solid #000;">MDLM (Ours)</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">95.26</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">32.83</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">67.01</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">47.52</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">61.15</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">41.89</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">37.37</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="fragment" data-fragment-index="1" style="font-size: 36px; line-height: 1.45; margin: 16px 0 8px 0; padding-left: 18px;">
  <div style="font-weight: 600; margin: 0 0 6px 0;">Datasets</div>
  <ul style="margin: 0; padding-left: 18px;">
    <li><strong>PTB:</strong> Related to Wall Street Journal</li>
    <li><strong>Lambada:</strong> English novels in BookCorpus.</li>
    <li><strong>PubMed:</strong> Citations and abstracts from biomedical literature.</li>
  </ul>
</div>

<div class="fragment" data-fragment-index="2" style="font-size: 36px; line-height: 1.45; margin: 12px 0 0 0; padding-left: 18px;">
  <div style="font-weight: 700; margin: 0 0 6px 0;">Takehome Message</div>
  <ul style="margin: 0; padding-left: 18px;">
    <li>MDLM outperforms SEDD on all datasets mentioned.</li>
    <li>Compared to AR models, MDLM shows better generalization ability.</li>
  </ul>
</div>
---

## Downstream NLP Tasks: GLUE Benchmark

<div style="font-size: 26px; line-height: 1.35; max-width: 1180px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 4: GLUE evaluation results. Evaluation measures (&uarr;) are F1 score for QQP and MRPC, Spearman correlations for STS-B, and accuracy for the rest. For MNLI, we report match/mismatch accuracies.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 32px; margin: 0 auto;">
    <thead>
      <tr>
        <th style="width: 16%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">MNLI<br>(m/mm)</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">QQP</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">QNLI</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">SST-2</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">COLA</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">STS-B</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">MRPC</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">RTE</th>
        <th style="text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Avg</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border-bottom: 1px solid #000;">AR</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">80.94/80.78</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">86.98</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">86.16</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">90.14</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">33.43</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">84.32</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">83.88</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">47.29</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">74.88</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">BERT</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">84.43/85.35</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">88.41</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">90.46</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">92.20</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">54.81</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">88.41</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">89.16</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">61.37</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">81.62</td>
      </tr>
      <tr>
        <td style="border-bottom: 2px solid #000;">BERT+MDLM-FT</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">84.76/85.07</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">88.49</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">90.30</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">92.20</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">57.69</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">87.48</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">90.53</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">62.09</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">82.06</td>
      </tr>
    </tbody>
  </table>
</div>

**Finetune Details:**
+ Using C4 dataset (Colossal Clean Crawled Corpus)
+ 5000 steps of generative fine-tuning with MDLM objective

---

## Comparing with Semi-AR Models

<div style="font-size: 28px; line-height: 1.35; max-width: 760px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 5: Semi-AR generative perplexity (Gen. PPL; &darr;) for sequences of 2048 tokens.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 24px;">
    <thead>
      <tr>
        <th style="width: 35%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="width: 32.5%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Gen. PPL (&darr;)</th>
        <th style="width: 32.5%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Sec/Seq (&darr;)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border-bottom: 1px solid #000;">SSD-LM</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">35.43</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">2473.9</td>
      </tr>
      <tr>
        <td style="border-bottom: 2px solid #000;">MDLM (Ours)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">27.18</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">89.3</td>
      </tr>
    </tbody>
  </table>
</div>

**Dataset:** `OWT`

**Generative Perplexity:**
1. Sample same number of sequences from each model.
2. Use a GPT2 trained on `OWT` to compute perplexity of the sampled sequences.

---

## DNA Sequence Modelling: Mamba

<div style="font-size: 26px; line-height: 1.35; max-width: 780px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 6: Test perplexities (PPL; &darr;) of generative fine-tuning of the Caduceus MLM [50] on the HG38 reference genome. Best diffusion model values are bolded. Error bars indicate the difference between the maximum and minimum values across 5 random seeds used for fine-tuning.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 28px;">
    <thead>
      <tr>
        <th style="width: 52%; border-top: 2px solid #000; border-bottom: 1px solid #000;"></th>
        <th style="width: 22%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Params</th>
        <th style="width: 26%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">PPL (&darr;)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="font-style: italic;" rowspan="2">Autoregressive (Retrained)</td>
        <td style="text-align: center;">465K</td>
        <td style="text-align: center;">3.067 &plusmn; .010</td>
      </tr>
      <tr>
        <td style="text-align: center; border-bottom: 1px solid #000;">433K</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">3.153 &plusmn; .001</td>
      </tr>
      <tr>
        <td style="font-style: italic;" rowspan="2">Diffusion (Retrained)</td>
        <td style="text-align: center;">507K(Plaid)</td>
        <td style="text-align: center;">&le; 3.240 &plusmn; .005</td>
      </tr>
      <tr>
        <td style="text-align: center; border-bottom: 1px solid #000;">467K(SEDD)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">&le; 3.216 &plusmn; .003</td>
      </tr>
      <tr>
        <td style="font-style: italic; border-bottom: 2px solid #000;">MDLM (Ours)</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">467K</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">&le; 3.199 &plusmn; .010</td>
      </tr>
    </tbody>
  </table>
</div>

---

## DNA Sequence Downstream Tasks

<div style="font-size: 26px; line-height: 1.35; max-width: 1180px; margin: 0 auto;">
  <div style="margin: 0 0 12px 0;">
    Table 7: Genomic Benchmarks. Top-1 accuracy (&uarr;) across 5-fold cross-validation (CV) for a pre-trained AR Mamba and a pre-trained Caduceus model fine-tuned with different diffusion parameterizations. The best values per task are bolded and the second best are italicized. Error bars indicate the difference between the maximum and minimum values across 5 random seeds used for CV.
  </div>
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 22px;">
    <thead>
      <tr>
        <th style="width: 32%; border-top: 2px solid #000; border-bottom: 1px solid #000;">Model Fine-Tuning Objective<br>(Parameter Count)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Mamba AR<br>(465K)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Caduceus MLM<br>(467K)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Caduceus Plaid<br>(507K)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Caduceus SEDD<br>(467K)</th>
        <th style="width: 13.6%; text-align: center; border-top: 2px solid #000; border-bottom: 1px solid #000;">Caduceus MDLM (ours)<br>(467K)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border-bottom: 1px solid #000;">Mouse Enhancers</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.763 (&plusmn;0.008)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.810 (&plusmn;0.016)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.745 (&plusmn;0.079)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.784 (&plusmn;0.058)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.795 (&plusmn;0.029)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Coding vs. Intergenomic</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.897 (&plusmn;0.004)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.913 (&plusmn;0.003)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.908 (&plusmn;0.003)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.913 (&plusmn;0.005)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.913 (&plusmn;0.003)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human vs. Worm</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.967 (&plusmn;0.002)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.970 (&plusmn;0.002)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.971 (&plusmn;0.001)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.970 (&plusmn;0.003)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.970 (&plusmn;0.003)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human Enhancers Cohn</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.734 (&plusmn;0.027)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.737 (&plusmn;0.011)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.743 (&plusmn;0.010)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.746 (&plusmn;0.015)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.743 (&plusmn;0.016)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human Enhancer Ensembl</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.856 (&plusmn;0.003)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.907 (&plusmn;0.000)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.885 (&plusmn;0.003)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.905 (&plusmn;0.006)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.899 (&plusmn;0.004)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human Regulatory</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.861 (&plusmn;0.008)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.874 (&plusmn;0.003)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.868 (&plusmn;0.010)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.828 (&plusmn;0.037)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.868 (&plusmn;0.004)</td>
      </tr>
      <tr>
        <td style="border-bottom: 1px solid #000;">Human OCR Ensembl</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.806 (&plusmn;0.005)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 1px solid #000;">0.821 (&plusmn;0.004)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.820 (&plusmn;0.004)</td>
        <td style="text-align: center; border-bottom: 1px solid #000;">0.816 (&plusmn;0.008)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 1px solid #000;">0.823 (&plusmn;0.008)</td>
      </tr>
      <tr>
        <td style="border-bottom: 2px solid #000;">Human NonTATA Promoters</td>
        <td style="text-align: center; border-bottom: 2px solid #000;">0.926 (&plusmn;0.008)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 2px solid #000;">0.935 (&plusmn;0.014)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 2px solid #000;">0.935 (&plusmn;0.007)</td>
        <td style="text-align: center; font-style: italic; border-bottom: 2px solid #000;">0.935 (&plusmn;0.014)</td>
        <td style="text-align: center; font-weight: bold; border-bottom: 2px solid #000;">0.940 (&plusmn;0.007)</td>
      </tr>
    </tbody>
  </table>
</div>

---

# Thank You
