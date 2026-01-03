# Deep-learning-model-for-ultrashort-pulses-reconstruction-from-FROG-data

## Problem description

The phase-retrieval problem appears when measuring ultrashort laser pulses. These pulses last less than a picosecond (~10<sup>-12</sup> s), so no electronic measurement can directly capture the electric field of the pulse in the time or frequency domain. In recent years, optical techniques have advanced to the point of measuring on the attosecond scale (~10<sup>-18</sup> s).

We use the Frequency Resolved Optical Gating system to characterize ultrashort pulse duration and the optical setups is shown in Figure 1(a). 

![Experimental setup](Figure1(a).png)
![Second Harmonic Generation in lab](Figure1(b).png)

*Figure 1: (a) Experimental setup for measuring SHG-FROG traces.  (b) Second Harmonic Generation in lab.

The pulse is autocorrelated with a delayed copy of itself and passed through a second-harmonic generation crystal, which produces a signal proportional to the product of the two incident pulses. Variants of this setup exist, but the objective is always to send the crystal’s signal through a spectrometer and obtain the **SHG-FROG trace** of the incident pulse, which is

$$
I_{\text{ SHG FROG}}(\omega,\tau)
= \left| \int_{-\infty}^{\infty} E(t)E(t-\tau)e^{-i\omega t}dt \right|^{2}
$$

The FROG spectrogram (FROG trace) is a graph of intensity as a function of frequency ω and delay 
τ. The signal field from nonlinear interaction is easier to express in the time domain, so the typical expression for the FROG trace includes a Fourier transform:

$$I_{\text{FROG}}(\omega, \tau) = |E_{\text{sig}}(\omega, \tau)|^2 = |FT[E_{\text{sig}}(t, \tau)]|^2 = \left|\int_{-\infty}^{\infty} E_{sig}(t, \tau)e^{-i\omega t} dt\right|^2$$

The raw data we get from the FROG system does not directly give us the pulse duration. We need to reconstruct pulse profile first from raw data before conduct the pulse duration measurement. Conventionally, we use Principle Compoent Generalized Projections Algorithums to complete this reconstruction task. This algorithum is very time wasting and can not deal with the noisy or strange pulse situation. Here, use try to use Deep learning model to replace this traditional algorithum to reconstruct pulse profile from raw FROG data.



---

## Summary

In summary, the theory tells us that from a SHG-FROG trace we can recover the time-domain electric field that generated it. The electric field can be written as intensity and phase:

$$
E(t) = \sqrt{I(t)}*e^{i\phi(t)}.
$$

This electric field is passed through the experimental setup so that the SHG-FROG trace is obtained.

Here, what will be obtained from the Frequency-resolved Optical Gating (FROG) system is the SHG-FROG trace(raw experimental data), a two-dimensional array of $N  *  N$ elements, $I_{mn}$ where $m,n = 0, \ldots, N-1$; and our goal is to retrieve the 1D array of length 2N that represents the complex electric field of ultrashort pulse, with first N elements corresponding to the real part and the remaining N elements to the imaginary part. 


![Summary figures](Figure2.png)

*Figure 2: Goal—transform the (N  x  N) SHG-FROG trace into the \(2N\) values that define the time-domain electric field.*

This problem can be addressed using dedicated phase-retrieval algorithms such as **GPA**, **PIE**, or **COPRA**. 

---

## Problem from DNN perspective

The aim of this project is to use **deep neural networks** to solve the phase-retrieval problem. The network’s task is to **invert** the mapping from the \(N x N\) SHG-FROG trace (real numbers) to the \(2N\) real numbers representing the electric field’s real and imaginary parts.

![NN perspective](Figure3.png)

*Figure 3: Schematic of the deep neural network that performs pulse retrieval.*


---

## Reconstruction of ultrashort pulses using this model

![NN perspective](Figure4.png)

## Author

This code was developed by **Xiangfeng(William) Deng** as part of the summer research program at the **Rice University**.

For questions or comments, please contact: phoenix.william.d@gmail.com

---

## Reference

Tom Zahavy, Alex Dikopoltsev, Daniel Moss, Gil Ilan Haham, Oren Cohen, Shie Mannor, and Mordechai Segev, "Deep learning reconstruction of ultrashort pulses," Optica 5, 666-673 (2018)

No reference code provided by paper author


