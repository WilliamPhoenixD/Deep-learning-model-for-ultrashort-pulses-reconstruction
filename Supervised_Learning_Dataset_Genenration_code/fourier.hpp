/**
 * @file fourier.hpp
 * @author Víctor Loras Herrero
 * @brief Functions related to the numerical computing of the Fourier Transform using FFTW library (https://www.fftw.org/)
 *
 * @copyright Copyright (c) 2023
 *
 * This module implements the Fourier transforms on linear grids.
 * Check out Nils C Geib "PyPret" Fourier module which inspired this code.
 * https://pypret.readthedocs.io/en/latest/apidoc/pypret.fourier.html
 *
 * Choosing the Fourier Transform definition as:
 *
 *      Ẽ(ω) = 1/2π ∫E(t)e^{i ω t} dt    ;    E(t) = ∫Ẽ(ω)e^{-i t ω} dω
 *
 * And discretizing it, the nth and the jth coefficient (direct and inverse transforms) will be:
 *
 *      Ẽ(ωₙ) := Ẽₙ = 1/2π ∑ⱼ₌₀ᴺ⁻¹ E(tⱼ) e^{i ωₙ tⱼ} Δt    ;     E(tⱼ) := Eⱼ = ∑ₙ₌₀ᴺ⁻¹ Ẽ(ωₙ) e^{-i tⱼ ωₙ} Δω
 *
 * Where tⱼ is the jth element of the time array and ωₙ is the nth element of the frequency array.
 *
 * The time array will be of the form: tⱼ = t₀ + j·Δt with j = 0, ..., N - 1, where N-1 is the number of samples.
 * The frequency array will be of the form ωₙ = ω₀ + n·Δω with n = 0, ..., N - 1.
 *
 * Considering the reciprocity relation, which states that:
 *
 *      Δt Δω = 2π/N
 *
 * We can substitute it into the obtained discretized expressions:
 *
 *      Ẽₙ = 1/2π ∑ⱼ₌₀ᴺ⁻¹ Eⱼ e^{i (ω₀ + n·Δω) (t₀ + j·Δt)} Δt =
 *      = Δt/2π e^{i n t₀ Δω} ∑ⱼ₌₀ᴺ⁻¹ Eⱼ e^{i ω₀ tⱼ} e^{i n j Δω Δt} =
 *      = Δt/2π e^{i n t₀ Δω} ∑ⱼ₌₀ᴺ⁻¹ Eⱼ e^{i ω₀ tⱼ} e^{i 2π n j / N}
 *
 *      Eⱼ = ∑ₙ₌₀ᴺ⁻¹ Ẽₙ e^{-i (t₀ + j·Δt) (ω₀ + n·Δω)} Δω =
 *      = Δω e^{-i ω₀ tⱼ} ∑ₙ₌₀ᴺ⁻¹ Ẽₙ e^{-i n t₀ Δω} e^{-i n j Δt Δω} =
 *      = Δω e^{-i ω₀ tⱼ} ∑ₙ₌₀ᴺ⁻¹ Ẽₙ e^{-i n t₀ Δω} e^{-i 2π n j / N} =
 *      = Δω e^{-i ω₀ tⱼ} ∑ₙ₌₀ᴺ⁻¹ Ẽₙ e^{-i n t₀ Δω} e^{-i 2π n j / N}
 *
 * If we define:
 *
 *      rₙ = e^{i n t₀ Δω} ; sⱼ = e^{i ω₀ tⱼ}
 *
 * We can finally express the Discrete Fourier Transform (DFT) as:
 *
 *      Ẽₙ = Δt/2π · rₙ · ∑ⱼ₌₀ᴺ⁻¹ Eⱼ·sⱼ e^{i 2π n j / N}    ;    Eⱼ = Δω · sⱼ* · ∑ₙ₌₀ᴺ⁻¹ Ẽₙ·rₙ* e^{-i 2π n j / N}
 *
 * So we can denote:
 *
 *      DFTₙ = ∑ⱼ₌₀ᴺ⁻¹ Eⱼ' e^{i 2π n j / N}    ;    IDFTⱼ = ∑ₙ₌₀ᴺ⁻¹ Ẽₙ' e^{-i 2π n j / N}
 *
 * (Where Eⱼ' = Eⱼ·sⱼ and Ẽₙ' = Ẽₙ·rₙ*)
 * So this definitions match the ones used in the fftw3 library for the forward and inverse transform.
 * Note that our forward transform matches the backwards transform in fftw3 and vice versa.
 * Also, remember that ifft in fftw3 does not have the 1/N amplitude factor.
 *
 *
 * Therefore, we can use the fast Fourier transform to compute the coefficients, yielding the following expressions:
 *
 *      Ẽₙ = Δt/2π · rₙ · ifft(Eⱼ·sⱼ)    ;    Eⱼ = Δω · sⱼ* · fft(Ẽₙ·rₙ*)
 *
 * And we don't have to worry about shifting the result of the fft.
 *
 * We should consider the Nyquist sampling theorem, which states that to avoid aliasing effects, we should discard all
 * frequencies higher than half the sampling frequency, given by fₘ = 1 / Δt. Thus, the frequency array should be
 * evenly spaced between -ωₘ/2 = -π/Δt and ωₘ/2 = π/Δt with Δω = 2π/(NΔt). If the frequency array does not meet this
 * relationship, we will have problems when switching between domains.
 *
 * Once again, check the Fourier module in PyPret for a more detailed discussion.
 * https://pypret.readthedocs.io/en/latest/apidoc/pypret.fourier.html
 */

#ifndef FOURIER_INCLUDED
#define FOURIER_INCLUDED
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <complex>

#include <fftw3.h>

/**
 * @brief Class for efficient computing of forward and backwards Fourier Transforms using FFTW.
 *
 * The idea is to provide an efficient way to compute a lot of forward and backward Fourier Transforms
 * for a vector defined on a fixed time and frequency grid with a given number of samples, N.
 *
 * FFTW generates a 'plan' instance that makes efficient computations of the fft for a fixed N,
 * so this class manages to mantain one of these plans to make efficient computations with the
 * matching phase factors given by the time and frequency arrays.
 *
 */
class FourierTransform
{
private:
    std::vector<std::complex<double>> r_n;      // Precomputed r_n phase factors
    std::vector<std::complex<double>> s_j;      // Precomputed s_j phase factors
    std::vector<std::complex<double>> r_n_conj; // Precomputed r_n phase factors conjugated
    std::vector<std::complex<double>> s_j_conj; // Precomputed s_j phase factors conjugated
    fftw_complex *in;                           // In array for the FFTW plan
    fftw_complex *out;                          // Out array for the FFTW plan
    fftw_plan forwardPlan;                      // FFTW plan for forward transform
    fftw_plan backwardPlan;                     // FFTW plan for backward transform

public:
    int N;                                    // Number of samples
    double deltaT;                            // Spacing in the time grid
    double deltaOmega;                        // Spacing in the angular frequency grid
    std::vector<double> t;                    // Time vector
    std::vector<double> omega;                // Angular frequency vector
    std::vector<std::complex<double>> result; // Array that stores the last result obtained

    /**
     * \brief Class constructor for a FourierTransform object.
     *
     * \param nSamples Number of samples of the vector to compute its transforms.
     * \param dt Grid spacing in the time domain of the given vector.
     * \param t0 Starting time of the time array in which the complex vector is defined.
     */
    FourierTransform(int nSamples, double dt, double t0)
    {
        this->N = nSamples;
        this->deltaT = dt;
        this->deltaOmega = 2 * M_PI / (N * deltaT);

        // Calculate the time and angular frequency grids
        t.resize(N);
        omega.resize(N);
        double start = -M_PI / deltaT;
        for (int i = 0; i < N; ++i)
        {
            t[i] = t0 + i * deltaT;
            omega[i] = start + (2 * M_PI * i / (N * deltaT));
        }

        // Compute r_n factors
        r_n.resize(N);
        r_n_conj.resize(N);
        // If starting time is zero, the phase factors are easier to compute.
        if (t[0] == 0.0)
        {
            for (int i = 0; i < N; i++)
            {
                r_n[i] = this->deltaT / (2 * M_PI); // We include here the amplitude factor in the forward transform for computational efficiency
                r_n_conj[i] = 1;
            }
        }
        else
        {
            double constFactor = t[0] * deltaOmega;
            for (int i = 0; i < N; i++)
            {
                r_n[i] = this->deltaT / (2 * M_PI) * std::exp(std::complex<double>(0, i * constFactor));
                r_n_conj[i] = std::exp(std::complex<double>(0, -i * constFactor));
            }
        }

        // Compute s_j factors
        s_j.resize(N);
        s_j_conj.resize(N);
        // If central frequency is zero, the phase factors are easier to compute.
        if (omega[0] == 0.0)
        {
            for (int i = 0; i < N; i++)
            {
                s_j[i] = 1;
                s_j_conj[i] = this->deltaOmega; // We include here the amplitude factor in the backward transform for computational efficiency
            }
        }
        else
        {
            for (int i = 0; i < N; i++)
            {
                s_j[i] = std::exp(std::complex<double>(0, t[i] * omega[0]));
                s_j_conj[i] = this->deltaOmega * std::exp(std::complex<double>(0, -t[i] * omega[0]));
            }
        }

        // Initialize FFTW plans
        in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
        out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
        forwardPlan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        backwardPlan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

        result.resize(N);
    }

    // Destructor
    ~FourierTransform()
    {
        fftw_free(in);
        fftw_free(out);
        fftw_destroy_plan(forwardPlan);
        fftw_destroy_plan(backwardPlan);
    }

    /**
     * \brief Computes the Discrete Fourier Transform (DFT) of a dataset.
     *
     * \param x The vector representing the complex valued dataset.
     * \return The vector representing the DFT of the dataset.
     */
    std::vector<std::complex<double>> forwardTransform(const std::vector<std::complex<double>> &x)
    {
        std::vector<std::complex<double>> x_sj(N);

        // Apply s_j factors to input vector
        for (int i = 0; i < N; i++)
        {
            x_sj[i] = x[i] * s_j[i];
        }

        // Prepare the input data
        for (int i = 0; i < N; i++)
        {
            in[i][0] = std::real(x_sj[i]);
            in[i][1] = std::imag(x_sj[i]);
        }

        // Execute the forward transform
        fftw_execute_dft(backwardPlan, in, out);

        // Process the output with the necessary factors
        for (int i = 0; i < N; i++)
        {
            result[i] = r_n[i] * std::complex<double>(out[i][0], out[i][1]);
        }

        return result;
    }

    /**
     * \brief Computes the Inverse Discrete Fourier Transform (IDFT) of a dataset.
     *
     * \param x The vector representing the complex valued dataset.
     * \return The vector representing the IDFT of the dataset.
     */
    std::vector<std::complex<double>> backwardTransform(const std::vector<std::complex<double>> &x)
    {
        std::vector<std::complex<double>> x_rn(N);
        for (int i = 0; i < N; i++)
        {
            x_rn[i] = x[i] * r_n_conj[i];
        }
        // Prepare the input data
        for (int i = 0; i < N; ++i)
        {
            in[i][0] = std::real(x_rn[i]); // Real part of the input
            in[i][1] = std::imag(x_rn[i]); // Imaginary part of the input
        }

        // Execute the backward transform
        fftw_execute_dft(forwardPlan, in, out);

        // Process the output with the necessary factors
        for (int i = 0; i < N; i++)
        {
            result[i] = s_j_conj[i] * std::complex<double>(out[i][0], out[i][1]);
        }

        return result;
    }
};

/**
 * Generate the frequency array for the discrete Fourier transform (DFT).
 * The function returns an equispaced frequency array corresponding to the given
 * number of samples and sample spacing.
 *
 * \param N The number of samples in the time domain.
 * \param deltaT The sample spacing (time interval) between consecutive samples.
 * \return A vector of frequencies in the frequency domain.
 */
std::vector<double> fftFreq(int N, double deltaT)
{
    std::vector<double> frequencies(N);
    double fSample = 1.0 / deltaT;
    double start = -0.5 * fSample;

    for (int i = 0; i < N; i++)
    {
        frequencies[i] = start + (i / (N * deltaT));
    }

    return frequencies;
}

/**
 * Converts a vector of frequencies to angular frequency units.
 *
 * @param frequencies The vector of frequencies to convert.
 * @return The vector of frequencies in angular frequency units.
 */
std::vector<double> toAngularFrequency(const std::vector<double> &frequencies)
{
    std::vector<double> angularFrequencies(frequencies.size());

    for (size_t i = 0; i < frequencies.size(); i++)
    {
        angularFrequencies[i] = 2 * M_PI * frequencies[i];
    }

    return angularFrequencies;
}

#endif // FOURIER_HPP