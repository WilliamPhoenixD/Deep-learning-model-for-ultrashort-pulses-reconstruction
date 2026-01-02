/**
 * @file retrievers.hpp
 * @author Víctor Loras Herrero
 * @brief Retriever classes for ultrashort pulse retrieval
 *
 * @copyright Copyright (c) 2023
 *
 * This module implements some retrieval methods for SHG-FROG traces, including:
 *      - Generalized Projections Algorithm (GPA)
 *      - Ptycographic Iterative Engine (PIE)
 *      - Common Phase Retrieval Algorithm (COPRA)
 *
 * Check out Nils C Geib "PyPret" module which inspired this code.
 * https://pypret.readthedocs.io/en/latest/
 *
 */

#ifndef RETRIEVERS_INCLUDED
#define RETRIEVERS_INCLUDED
#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
#include <vector>
#include "fourier.hpp"
#include "pulse.hpp"

/**
 * @class retrieverBase
 * @brief Base class for pulse retrieval algorithms.
 *
 * This class provides the basic structure and variables needed for pulse retrieval algorithms.
 */
class retrieverBase
{
public:
    FourierTransform *_ft; // Fourier transform object to perform fast fourier transforms

    int N; // Number of samples

    Pulse *result; // Resulting pulse of the retrieval

    std::vector<std::vector<double>> Tmeas; // Measured trace
    double TmeasMaxSquared;                 // Maximum value of the measured trace

    std::vector<double> tau;                               // Delays of the pulse in the time domain
    std::vector<std::vector<std::complex<double>>> delays; // Delays of the pulse in the frequency domain. NxN matrix that stores each delay for each frequency

    std::vector<std::vector<std::complex<double>>> Smn;     // Signal operator in frequency domain
    std::vector<std::vector<std::complex<double>>> Smk;     // Signal operator in time domain
    std::vector<std::vector<std::complex<double>>> nextSmk; // Signal operator in time domain after projection
    std::vector<std::vector<std::complex<double>>> Amk;     // Delayed pulse by τ_m
    std::vector<std::vector<double>> Tmn;                   // Trace of the resulting pulse

    double mu;        // Scale factor
    double r;         // Sum of squared residuals
    double R;         // Trace error
    double bestError; // Best achieved trace error

    std::vector<double> allTraceErrors; // This will store al retrieval errors during the retrieval process.

    double Z;                                // Sum of difference between the signal operators in the frequency domain
    std::vector<std::complex<double>> gradZ; // Stores the value of the gradient of Z
    double gamma;                            // Gradient descent step

    bool verbose = false; // Verbose mode

    retrieverBase(FourierTransform &ft, std::vector<std::vector<double>> Tmeasured)
    {

        this->_ft = &ft;        // Fourier transform object to perform fast fourier transforms
        this->N = this->_ft->N; // Number of samples

        //! Starting pulse as a random pulse with TBP = 0.5
        this->result = new Pulse(ft);   // Resulting pulse of the retrieval
        this->result->randomPulse(0.5); // Random pulse with TBP = 0.5

        this->Tmeas = Tmeasured; // Copy the measured trace
        this->TmeasMaxSquared = 0;
        for (int i = 0; i < this->N; i++)
        {
            for (int j = 0; j < this->N; j++)
            {
                if (Tmeasured[i][j] > this->TmeasMaxSquared)
                {
                    this->TmeasMaxSquared = Tmeasured[i][j];
                }
            }
        }

        this->TmeasMaxSquared *= this->TmeasMaxSquared; // Square the maximum value of the measured trace

        this->tau.reserve(this->N); // Delays of the pulse in the time domain
        for (int i = 0; i < this->N; i++)
        {
            this->tau[i] = (i - std::floor(0.5 * this->N)) * this->_ft->deltaT; // Delay values, given by the time step and the number of samples
        }

        this->delays.resize(this->N, std::vector<std::complex<double>>(this->N)); // Delays of the pulse in the frequency domain. NxN matrix that stores each delay for each frequency

        for (int i = 0; i < this->N; i++) // iterates through delay values
        {
            for (int j = 0; j < this->N; j++) // iterates through frequency values
            {
                this->delays[i][j] = std::exp(std::complex<double>(0, this->_ft->omega[j] * tau[i])); // delay in the time domain by τ. It is a phase factor, exp(iωτ). Note the positive sign.
            }
        }

        this->Smn.resize(this->N, std::vector<std::complex<double>>(this->N));     // Signal operator in frequency domain
        this->Smk.resize(this->N, std::vector<std::complex<double>>(this->N));     // Signal operator in time domain
        this->nextSmk.resize(this->N, std::vector<std::complex<double>>(this->N)); // Signal operator in time domain after projection
        this->Amk.resize(this->N, std::vector<std::complex<double>>(this->N));     // Delayed pulse by τ_m
        this->Tmn.resize(this->N, std::vector<double>(this->N));                   // SHG-FROG trace of the resulting pulse

        this->gradZ.reserve(this->N); // Stores the value of the gradient of Z. This varies depending on the algorithm.
    }

    retrieverBase(FourierTransform &ft, std::vector<std::vector<double>> Tmeasured, std::vector<std::complex<double>> &candidateField)
    {

        this->_ft = &ft;        // Fourier transform object to perform fast fourier transforms
        this->N = this->_ft->N; // Number of samples

        //! Starting pulse set by the user
        this->result = new Pulse(ft);   // Resulting pulse of the retrieval
        this->result->setField(candidateField); // Set the field of the pulse as the candidate field

        this->Tmeas = Tmeasured; // Copy the measured trace
        this->TmeasMaxSquared = 0;
        for (int i = 0; i < this->N; i++)
        {
            for (int j = 0; j < this->N; j++)
            {
                if (Tmeasured[i][j] > this->TmeasMaxSquared)
                {
                    this->TmeasMaxSquared = Tmeasured[i][j];
                }
            }
        }

        this->TmeasMaxSquared *= this->TmeasMaxSquared; // Square the maximum value of the measured trace

        this->tau.reserve(this->N); // Delays of the pulse in the time domain
        for (int i = 0; i < this->N; i++)
        {
            this->tau[i] = (i - std::floor(0.5 * this->N)) * this->_ft->deltaT; // Delay values, given by the time step and the number of samples
        }

        this->delays.resize(this->N, std::vector<std::complex<double>>(this->N)); // Delays of the pulse in the frequency domain. NxN matrix that stores each delay for each frequency

        for (int i = 0; i < this->N; i++) // iterates through delay values
        {
            for (int j = 0; j < this->N; j++) // iterates through frequency values
            {
                this->delays[i][j] = std::exp(std::complex<double>(0, this->_ft->omega[j] * tau[i])); // delay in the time domain by τ. It is a phase factor, exp(iωτ). Note the positive sign.
            }
        }

        this->Smn.resize(this->N, std::vector<std::complex<double>>(this->N));     // Signal operator in frequency domain
        this->Smk.resize(this->N, std::vector<std::complex<double>>(this->N));     // Signal operator in time domain
        this->nextSmk.resize(this->N, std::vector<std::complex<double>>(this->N)); // Signal operator in time domain after projection
        this->Amk.resize(this->N, std::vector<std::complex<double>>(this->N));     // Delayed pulse by τ_m
        this->Tmn.resize(this->N, std::vector<double>(this->N));                   // SHG-FROG trace of the resulting pulse

        this->gradZ.reserve(this->N); // Stores the value of the gradient of Z. This varies depending on the algorithm.
    }

    /**
     * @brief Destroy the retriever Base object
     *
     */
    ~retrieverBase()
    {
        delete this->result;
    }

    /**
     * @brief Retrieves the pulse from the measured trace.
     *
     * @param tolerance Tolerance of the retrieval. The retrieval will stop when the trace error is below this value.
     * @param maximumIterations Maximum number of iterations for the retrieval.
     * @return Pulse Retrieved pulse.
     */
    virtual Pulse retrieve(double tolerance, double maximumIterations) = 0; //! Pure virtual function. This function must be implemented in the derived classes.

    /**
     * @brief Computes the Amk matrix from the spectrum.
     *
     * Amk is the delayed pulse by τ_m. It is the spectrum of the pulse multiplied by the delay in the time domain by τ.
     *
     * @param spectrum Spectrum of the pulse.
     */
    void computeAmk(const std::vector<std::complex<double>> &spectrum)
    {
        std::vector<std::vector<std::complex<double>>> delayedSpectrum(this->N, std::vector<std::complex<double>>(this->N));
        for (int i = 0; i < this->N; i++) // iterates through delay values
        {
            for (int j = 0; j < this->N; j++) // iterates through frequency values
            {
                delayedSpectrum[i][j] = spectrum[j] * this->delays[i][j]; // delay in the time domain by τ
            }
            this->Amk[i] = this->_ft->backwardTransform(delayedSpectrum[i]); // E(t - τ)
        }
    }

    /**
     * @brief Only computes the Amk matrix for the given random index. This is convenient for PIE and COPRA.
     *
     * Amk is the delayed pulse by τ_m. It is the spectrum of the pulse multiplied by the delay in the time domain by τ.
     *
     * @param spectrum Spectrum of the pulse.
     * @param randomIndex Index of the random delay to compute.
     */
    void computeAmk(const std::vector<std::complex<double>> &spectrum, int randomIndex)
    {
        std::vector<std::complex<double>> delayedSpectrum(this->N);
        for (int j = 0; j < this->N; j++)
        {
            delayedSpectrum[j] = spectrum[j] * this->delays[randomIndex][j]; // delay in the time domain by τ
        }
        this->Amk[randomIndex] = this->_ft->backwardTransform(delayedSpectrum); // E(t - τ)
    }

    /**
     * @brief Computes the Smk matrix from the field.
     *
     * Smk is the signal operator in the time domain. It is the field multiplied by the delayed field by τ_m.
     * Sₘₖ = E(tₖ)·E(tₖ - τₘ),  m = 0, ... , M - 1 ; k = 0, ..., N - 1
     *
     * @param field Field of the pulse.
     */
    void computeSmk(const std::vector<std::complex<double>> &field)
    {
        for (int i = 0; i < this->N; i++)
        {
            for (int j = 0; j < this->N; j++)
            {
                this->Smk[i][j] = Amk[i][j] * field[j]; // E(t - τ) E(t)
            }
        }
    }

    /**
     * @brief Only computes the Smk matrix for the given random index. This is convenient for PIE and COPRA.
     *
     * Smk is the signal operator in the time domain. It is the field multiplied by the delayed field by τ_m.
     * Sₘₖ = E(tₖ)·E(tₖ - τₘ),  m = 0, ... , M - 1 ; k = 0, ..., N - 1
     *
     * @param field Field of the pulse.
     * @param randomIndex Index of the random delay to compute.
     */
    void computeSmk(const std::vector<std::complex<double>> &field, int randomIndex)
    {
        for (int j = 0; j < this->N; j++)
        {
            this->Smk[randomIndex][j] = Amk[randomIndex][j] * field[j]; // E(t - τ) E(t)
        }
    }

    /**
     * @brief Computes the Smn matrix from the Smk matrix.
     *
     * Smn is the signal operator in the frequency domain. It is the fourier transform of Smk.
     * Sₘₙ = ℱ{Sₘₖ²}
     *
     */
    void computeSmn()
    {
        for (int i = 0; i < this->N; i++)
        {
            this->Smn[i] = this->_ft->forwardTransform(this->Smk[i]);
        }
    }

    /**
     * @brief Only computes the Smn matrix for the given random index. This is convenient for PIE and COPRA.
     *
     * Smn is the signal operator in the frequency domain. It is the fourier transform of Smk.
     * Sₘₙ = ℱ{Sₘₖ²}
     *
     * @param randomIndex Index of the random delay to compute.
     */
    void computeSmn(int randomIndex)
    {

        this->Smn[randomIndex] = this->_ft->forwardTransform(this->Smk[randomIndex]);
    }

    /**
     * @brief Computes the Tmn matrix from the Smn matrix.
     *
     * Tmn is the SHG-FROG trace of the pulse. It is the square of the absolute value of Smn.
     * Tₘₙ = |Sₘₙ|² = ℱ{|Sₘₖ|²}
     *
     */
    void computeTmn()
    {
        for (int i = 0; i < this->N; i++)
        {
            for (int j = 0; j < this->N; j++)
            {
                this->Tmn[i][j] = std::norm(this->Smn[i][j]);
            }
        }
    }

    /**
     * @brief Only computes the Tmn matrix for the given random index. This is convenient for PIE and COPRA.
     *
     * Tmn is the SHG-FROG trace of the pulse. It is the square of the absolute value of Smn.
     * Tₘₙ = |Sₘₙ|² = ℱ{|Sₘₖ|²}
     *
     * @param randomIndex Index of the random delay to compute.
     */
    void computeTmn(int randomIndex)
    {
        for (int j = 0; j < this->N; j++)
        {
            this->Tmn[randomIndex][j] = std::norm(this->Smn[randomIndex][j]);
        }
    }

    /**
     * @brief Computes the scale factor mu.
     *
     * mu is the scale factor that minimizes the sum of squared residuals. Given by the expression:
     * μ = ∑ₘₙ (Tₘₙᵐᵉᵃˢ · Tₘₙ) / (∑ₘₙ Tₘₙ²)
     *
     */
    void computeMu()
    {
        double sum_meas_candidate = 0;
        double sum_meas = 0;
        for (int i = 0; i < this->N; i++)
        {
            for (int j = 0; j < this->N; j++)
            {
                sum_meas_candidate += this->Tmeas[i][j] * this->Tmn[i][j];
                sum_meas += this->Tmn[i][j] * this->Tmn[i][j];
            }
        }

        this->mu = sum_meas_candidate / sum_meas;
    }

    /**
     * @brief Computes the sum of squared residuals.
     *
     * r is the sum of squared residuals. Given by the expression:
     * r = ∑ₘₙ (Tₘₙᵐᵉᵃˢ - μ · Tₘₙ)²
     *
     */
    void computeResiduals()
    {
        std::vector<std::vector<double>> difference(this->N, std::vector<double>(this->N));
        double sum = 0.0;

        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < this->N; ++j)
            {
                difference[i][j] = this->Tmeas[i][j] - this->mu * this->Tmn[i][j]; // Tmeas - mu * Tmn
                sum += difference[i][j] * difference[i][j];                        // (Tmeas - mu * Tmn)²
            }
        }

        this->r = sum;
    }

    /**
     * @brief Computes the trace error.
     *
     * R is the trace error. Given by the expression:
     *  R = r½ / [M·N (maxₘₙ Tₘₙᵐᵉᵃˢ)²]½
     *
     */
    void computeTraceError()
    {
        this->R = sqrt(this->r / (this->N * this->N * this->TmeasMaxSquared));
    }

    /**
     * @brief Sets the initial field of the pulse that will be used for the retrieval.
     *
     * @param initialField Initial field of the pulse.
     */
    void setInitialField(const std::vector<std::complex<double>> &initialField)
    {
        this->result->setField(initialField); // Updates also the spectrum!
    }

    /**
     * @brief Sets the initial spectrum of the pulse that will be used for the retrieval.
     *
     * @param initialSpectrum Initial spectrum of the pulse.
     */
    void setInitialSpectrum(const std::vector<std::complex<double>> &initialSpectrum)
    {
        this->result->setSpectrum(initialSpectrum); // Updates also the field!
    }
};

/**
 * @brief Generalized Projections Algorithm (GPA) for pulse retrieval of SHG-FROG traces.
 *
 * This class implements the GPA algorithm for pulse retrieval of SHG-FROG traces.
 * It consists of the following steps:
 *
 *     - Step 1: Projection onto Sₘₖ. The new value of the candidate pulse signal operator, S'ₘₖ, is calculated by
 *                   performing a projection onto the set of pulses that satisfy Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, so the following
 *                   projection is performed: S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
 *
 *   - Step 2: Updating the electric field, E, through a gradient descent. For this, we define Z as the
 *               distance between S'ₘₖ and Sₘₖ, that is, Z = ∑ₘₖ |S'ₘₖ - Sₘₖ|². In this way, the gradient descent
 *               will be given by Eⱼ' = Eⱼ - γ·∇Zⱼ ; where γ is a control of the descent step. In the GPA algorithm
 *               a linear search is usually performed to find it, but a faster and equally valid option is to take γ = Z / ∑ⱼ|∇Zⱼ|².
 *
 *   - Step 3: Calculation of the new parameters for the next iteration and error in the pulse trace. The new values of the signal operator and the trace of the candidate pulse obtained by gradient descent
 *               in step 2 are calculated. The trace error R is calculated, and if the convergence condition is satisfied or the maximum number of iterations is reached, the algorithm stops. Otherwise, return to step 1.
 *
 */
class GPA : public retrieverBase
{
private:
    std::vector<std::complex<double>> bestField; // Result of the best field for retrieval

    /**
     * @brief Computes the next value of the signal operator in the time domain after projection.
     *
     * The projection is given by:
     * S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
     *
     */
    void computeNextSmk()
    {
        std::vector<std::vector<double>> absSmn(this->N, std::vector<double>(this->N)); // Stores the absolute value of Smn
        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < this->N; ++j)
            {
                absSmn[i][j] = std::abs(this->Smn[i][j]);
            }
        }

        std::vector<std::complex<double>> nextSmn(this->N);
        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < this->N; ++j)
            {
                if (absSmn[i][j] > 0.0)
                {
                    nextSmn[j] = this->Smn[i][j] / absSmn[i][j] * sqrt(this->Tmeas[i][j]); // Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ
                }
                else
                {
                    nextSmn[j] = sqrt(this->Tmeas[i][j]);
                }
            }

            this->nextSmk[i] = this->_ft->backwardTransform(nextSmn); // S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
        }
    }

    /**
     * @brief Computes the value of gamma, the step of the gradient descent.
     *
     * In the GPA algorithm a linear search is usually performed to find it, but a faster and equally valid option is to take:
     * γ = Z / ∑ⱼ|∇Zⱼ|²
     *
     */
    void computeGamma()
    {
        double sumGradZ = 0; // Sum of the absolute value of the gradient of Z
        for (int i = 0; i < this->N; i++)
        {
            sumGradZ += std::abs(this->gradZ[i]) * std::abs(this->gradZ[i]); // ∑ⱼ|∇Zⱼ|²
        }

        this->gamma = this->Z / sumGradZ; // γ = Z / ∑ⱼ|∇Zⱼ|²
    }

    /**
     * @brief Computes the gradient of Z. Given by:
     *
     * ∇Z = -2 ∑ₘ(S'ₘⱼ·E*₍ⱼ₊ₘ₎ - Sₘⱼ·E*₍ⱼ₊ₘ₎) + (S'₍ⱼ₋ₘ₎ⱼ·E*₍ⱼ₋ₘ₎ - S₍ⱼ₋ₘ₎ⱼ·E*₍ⱼ₋ₘ₎)
     *    = -2 ∑ₘ ΔSₘⱼ·E*₍ⱼ₊ₘ₎ + ΔS₍ⱼ₋ₘ₎ⱼ·E*₍ⱼ₋ₘ₎
     *
     * Assuming periodic continuation of the field
     *
     */
    void computeGradient()
    {
        // Calculate dS
        std::vector<std::vector<std::complex<double>>> dS(this->N, std::vector<std::complex<double>>(this->N));
        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < this->N; ++j)
            {
                dS[i][j] = this->nextSmk[i][j] - this->Smk[i][j]; //! This can be computed first in Z and save some time if stored
            }
        }

        for (int j = 0; j < this->N; j++)
        {
            this->gradZ[j] = 0;

            for (int m = 0; m < this->N; m++)
            {
                this->gradZ[j] += dS[m][j] * std::conj(this->Amk[m][j]); // ΔSₘⱼ·E*₍ⱼ₊ₘ₎
                if (j + m < N)                                           // Periodic continuation of the field
                {
                    this->gradZ[j] += dS[m][j + m] * std::conj(this->Amk[m][j + m]); // ΔS₍ⱼ₋ₘ₎ⱼ·E*₍ⱼ₋ₘ₎
                }
                else
                {
                    this->gradZ[j] += dS[m][j + m - this->N] * std::conj(this->Amk[m][j + m - this->N]); // ΔS₍ⱼ₋ₘ₎ⱼ·E*₍ⱼ₋ₘ₎
                }
            }

            this->gradZ[j] *= -2; // -2 ∑ₘ ΔSₘⱼ·E*₍ⱼ₊ₘ₎ + ΔS₍ⱼ₋ₘ₎ⱼ·E*₍ⱼ₋ₘ₎
        }
    }

    /**
     * @brief Computes the next value of the electric field after the gradient descent.
     *
     * E'ₖ = Eₖ - γ·∇Zₖ
     *
     */
    void computeNextField()
    {
        std::vector<std::complex<double>> currentField = this->result->getField();
        for (int i = 0; i < this->N; i++)
        {
            currentField[i] -= this->gamma * this->gradZ[i]; // E'ₖ = Eₖ - γ·∇Zₖ
        }

        this->result->setField(currentField); // Updates also the spectrum!
    }

    /**
     * @brief Computes the value of Z, the sum of difference between the signal operators in the frequency domain.
     *
     * Z = ∑ₘⱼ |S'ₘⱼ - Sₘⱼ|² = ∑ₘⱼ ΔSₘⱼ
     */
    void computeZ()
    {
        this->Z = 0;

        for (int i = 0; i < this->N; i++)
        {
            for (int j = 0; j < this->N; j++)
            {
                this->Z += norm(this->nextSmk[i][j] - this->Smk[i][j]); // ∑ⱼ |S'ₘⱼ - Sₘⱼ|²
            }
        }
    }

public:
    /**
     * @brief Construct a new GPA object. Inherits constructor from retrieverBase.
     *
     * @param ft Fourier transform object to perform fast fourier transforms
     * @param Tmeasured Measured trace
     */
    GPA(FourierTransform &ft, std::vector<std::vector<double>> Tmeasured) : retrieverBase(ft, Tmeasured)
    {
    }

    /**
     * @brief Construct a new GPA object. Inherits constructor from retrieverBase.
     *
     * @param ft Fourier transform object to perform fast fourier transforms
     * @param Tmeasured Measured trace
     * @param measuredDelays Delays of the experimentally measured trace
     */
    GPA(FourierTransform &ft, std::vector<std::vector<double>> Tmeasured, std::vector<double> measuredDelays) : retrieverBase(ft, Tmeasured)
    {
        // Set up the delays by the given measured delays
        for (int i = 0; i < this->N; i++) // iterates through delay values
        {
            for (int j = 0; j < this->N; j++) // iterates through frequency values
            {
                this->delays[i][j] = std::exp(std::complex<double>(0, this->_ft->omega[j] * measuredDelays[i])); // delay in the time domain by τ
            }
        }
    }

    /**
     * @brief Retrieves the pulse from the measured trace.
     *
     * It consists of the following steps:
     *
     *     - Step 1: Projection onto Sₘₖ. The new value of the candidate pulse signal operator, S'ₘₖ, is calculated by
     *               performing a projection onto the set of pulses that satisfy Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, so the following
     *               projection is performed: S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
     *
     *      - Step 2: Updating the electric field, E, through a gradient descent. For this, we define Z as the
     *               distance between S'ₘₖ and Sₘₖ, that is, Z = ∑ₘₖ |S'ₘₖ - Sₘₖ|². In this way, the gradient descent
     *               will be given by Eⱼ' = Eⱼ - γ·∇Zⱼ ; where γ is a control of the descent step. In the GPA algorithm
     *               a linear search is usually performed to find it, but a faster and equally valid option is to take γ = Z / ∑ⱼ|∇Zⱼ|².
     *
     *       - Step 3: Calculation of the new parameters for the next iteration and error in the pulse trace. The new values of the signal operator and the trace of the candidate pulse obtained by gradient descent
     *               in step 2 are calculated. The trace error R is calculated, and if the convergence condition is satisfied or the maximum number of iterations is reached, the algorithm stops. Otherwise, return to step 1.
     *
     * @param tolerance Tolerance of the retrieval. The retrieval will stop when the trace error is below this value.
     * @param maximumIterations Maximum number of iterations for the retrieval.
     * @return Pulse A Pulse class instance containing the resulting pulse of the retrieval.
     */
    Pulse retrieve(double tolerance, double maximumIterations)
    {

        int nIter = 0;                                             // Number of iterations
        this->bestError = std::numeric_limits<double>::infinity(); // Best error is set to infinity
        this->computeAmk(this->result->getSpectrum());             // Compute Amk from the spectrum
        this->computeSmk(this->result->getField());                // Compute Smk from the field
        this->computeSmn();                                        // Compute Smn from Smk
        this->computeTmn();                                        // Compute Tmn from Smn
        this->computeMu();                                         // Compute mu, depends on Tmn
        this->computeResiduals();                                  // Compute residuals, depends on mu and Tmn
        this->computeTraceError();                                 // Compute trace error, depends on residuals
        this->allTraceErrors.push_back(this->R);                   // Store the first value of the trace error

        while (this->R > tolerance && nIter < maximumIterations) // While the trace error is above the tolerance and the maximum number of iterations is not reached
        {
            this->computeNextSmk(); // Compute the next value of the signal operator in the time domain after projection

            this->computeZ();        // Compute the value of Z, the sum of difference between the signal operators in the frequency domain
            this->computeGradient(); // Compute the gradient of Z
            this->computeGamma();    // Compute the value of gamma, the step of the gradient descent

            this->computeNextField(); // Compute the next value of the electric field after the gradient descent

            this->computeAmk(this->result->getSpectrum()); // Compute Amk from the spectrum
            this->computeSmk(this->result->getField());    // Compute Smk from the field
            this->computeSmn();                            // Compute Smn from Smk
            this->computeTmn();                            // Compute Tmn from Smn
            this->computeMu();                             // Compute mu, depends on Tmn
            this->computeResiduals();                      // Compute residuals, depends on mu and Tmn
            this->computeTraceError();                     // Compute trace error, depends on residuals
            this->allTraceErrors.push_back(this->R);       // Store the trace error

            if (this->R < this->bestError) // If the current trace error is the best
            {
                this->bestError = this->R;                  // Update the best error
                this->bestField = this->result->getField(); // Update the best field
            }

            if (this->verbose)
            {
                std::cout << "Iteration = " << nIter + 1 << "\t"
                          << "R = " << this->R << std::endl;
            }

            nIter++;
        }

        std::cout << "Best retrieval error R = " << this->bestError << std::endl;

        this->result->setField(this->bestField); // Set the best field as the result of the retrieval. Updates also the spectrum!

        this->allTraceErrors.push_back(this->bestError); //! The last value of the array is the best result. Not the last retrieval result.

        return *this->result;
    }
};

/**
 * @brief Ptychographic Iterative Engine (PIE) for pulse retrieval of SHG-FROG traces.
 */
class PIE : public retrieverBase
{
private:
    std::vector<std::complex<double>> bestField; // Result of the best field for retrieval

    /**
     * @brief Computes the next value of the signal operator in the time domain after projection.
     *
     * The projection is given by:
     * S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
     *
     */
    void computeNextSmk()
    {
        std::vector<std::vector<double>> absSmn(this->N, std::vector<double>(this->N));
        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < this->N; ++j)
            {
                absSmn[i][j] = std::abs(this->Smn[i][j]); // |Sₘₙ|
            }
        }

        std::vector<std::complex<double>> nextSmn(this->N);
        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < this->N; ++j)
            {
                if (absSmn[i][j] > 0.0)
                {
                    nextSmn[j] = this->Smn[i][j] / absSmn[i][j] * sqrt(this->Tmeas[i][j] / this->mu); // Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ
                }
                else
                {
                    nextSmn[j] = sqrt(this->Tmeas[i][j] / this->mu); // Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ
                }
            }

            this->nextSmk[i] = this->_ft->backwardTransform(nextSmn); // S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
        }
    }

    /**
     * @brief Computes the next value of the signal operator in the time domain after projection.
     *
     * Only computes the next value of the signal operator in the time domain after projection for the given random index.
     *
     * The projection is given by:
     * S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
     *
     */
    void computeNextSmk(int randomIndex)
    {
        std::vector<double> absSmn(this->N);
        for (int j = 0; j < this->N; ++j)
        {
            absSmn[j] = std::abs(this->Smn[randomIndex][j]);
        }

        std::vector<std::complex<double>> nextSmn(this->N);
        for (int j = 0; j < this->N; ++j)
        {
            if (absSmn[j] > 0.0)
            {
                nextSmn[j] = this->Smn[randomIndex][j] / absSmn[j] * sqrt(this->Tmeas[randomIndex][j] / this->mu); // Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ
            }
            else
            {
                nextSmn[j] = sqrt(this->Tmeas[randomIndex][j] / this->mu); // Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ
            }
        }

        this->nextSmk[randomIndex] = this->_ft->backwardTransform(nextSmn); // S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
    }

    /**
     * @brief Computes the value of the next field using the updated value of the signal operator in the time domain for the given random index.
     *
     * E'ₖ = Eₖ + β·∑ₖ ΔSₘₖ·E*ₖ₊ₘ / maxₖ |Eₖ|²
     *
     * Where β is a control parameter, m is the random index, and ΔSₘₖ = S'ₘₖ - Sₘₖ
     *
     * @param randomIndex Index of the random delay to compute.
     * @param beta Control parameter of the PIE algorithm.
     */
    void computeNextField(int randomIndex, double beta)
    {
        this->computeAmk(this->result->getSpectrum(), randomIndex); // Compute Amk from the spectrum
        this->computeSmk(this->result->getField(), randomIndex);    // Compute Smk from the field
        this->computeSmn(randomIndex);                              // Compute Smn from Smk
        this->computeTmn(randomIndex);                              // Compute Tmn from Smn

        // Compute projection on Smk
        this->computeNextSmk(randomIndex);

        std::vector<std::complex<double>> currentField = this->result->getField(); // Get the current field

        // Pick the maximum absolute value of the field
        double currentAbsMaxValue = 0;
        double absValue;

        currentAbsMaxValue = 0;
        for (int j = 0; j < this->N; j++)
        {
            absValue = std::norm(currentField[j]);
            if (currentAbsMaxValue < absValue)
            {
                currentAbsMaxValue = absValue;
            }
        }

        // Update the field, by doing : E'ₖ = Eₖ + β·∑ₖ ΔSₘₖ·E*ₖ₊ₘ / maxₖ |Eₖ|²
        // where β is a control parameter, m is the random index, and ΔSₘₖ = S'ₘₖ - Sₘₖ
        for (int j = 0; j < this->N; j++)
        {
            currentField[j] += beta * std::conj(this->Amk[randomIndex][j]) * (this->nextSmk[randomIndex][j] - this->Smk[randomIndex][j]) / currentAbsMaxValue;
        }

        this->result->setField(currentField); // Updates also the spectrum!
    }

    /**
     * @brief Computes an array of random indexes between 0 and N-1.
     *
     * @return std::vector<int> the array of random indexes.
     */
    std::vector<int> randomIndexShuffle()
    {
        std::vector<int> indices(this->N);
        std::random_device rd;
        std::mt19937 rng(rd());

        std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., N-1

        // Shuffle the array of indices
        std::shuffle(indices.begin(), indices.end(), rng);

        return indices;
    }

public:
    /**
     * @brief Construct a new PIE object. Inherits constructor from retrieverBase.
     *
     * @param ft Fourier transform object to perform fast fourier transforms
     * @param Tmeasured Measured trace
     */
    PIE(FourierTransform &ft, std::vector<std::vector<double>> Tmeasured) : retrieverBase(ft, Tmeasured)
    {
    }

    /**
     * @brief Construct a new PIE object. Inherits constructor from retrieverBase.
     *
     * @param ft Fourier transform object to perform fast fourier transforms
     * @param Tmeasured Measured trace
     * @param measuredDelays Delays of the experimentally measured trace
     */
    PIE(FourierTransform &ft, std::vector<std::vector<double>> Tmeasured, std::vector<double> measuredDelays) : retrieverBase(ft, Tmeasured)
    {
        // Set up the delays by the given measured delays
        for (int i = 0; i < this->N; i++) // iterates through delay values
        {
            for (int j = 0; j < this->N; j++) // iterates through frequency values
            {
                this->delays[i][j] = std::exp(std::complex<double>(0, this->_ft->omega[j] * measuredDelays[i])); // delay in the time domain by τ
            }
        }
    }

    /**
     * @brief Retrieves the pulse from the measured trace.
     *
     * PIE updates the field by applying the following expression for the array of signal operators updated in random order:
     *
     *  E'ₖ = Eₖ + β·∑ₖ ΔSₘₖ·E*ₖ₊ₘ / maxₖ |Eₖ|²
     *
     * Where β is a random number between 0.1 and 0.5, m is the random index in which the field is updated, and ΔSₘₖ = S'ₘₖ - Sₘₖ
     *
     * The projection is given by:
     * S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
     *
     * This projection is done for all the indexes in a random order. This has proven to give better results than doing the projection in a sequential order.
     *
     * @param tolerance  Tolerance of the retrieval. The retrieval will stop when the trace error is below this value.
     * @param maximumIterations  Maximum number of iterations for the retrieval.
     * @return Pulse  A Pulse class instance containing the resulting pulse of the retrieval.
     */
    Pulse retrieve(double tolerance, double maximumIterations)
    {

        int nIter = 0;                                             // Number of iterations
        this->bestError = std::numeric_limits<double>::infinity(); // Best error is set to infinity
        this->computeAmk(this->result->getSpectrum());             // Compute Amk from the spectrum
        this->computeSmk(this->result->getField());                // Compute Smk from the field
        this->computeSmn();                                        // Compute Smn from Smk
        this->computeTmn();                                        // Compute Tmn from Smn
        this->computeMu();                                         // Compute mu, depends on Tmn
        this->computeResiduals();                                  // Compute residuals, depends on mu and Tmn
        this->computeTraceError();                                 // Compute trace error, depends on residuals
        this->allTraceErrors.push_back(this->R);                   // Store the first value of the trace error

        std::vector<int> randomIndexes; // Array of random indexes

        // Set up a random number generator for beta
        std::random_device rd;                                         // obtain a random number from hardware
        std::mt19937 gen(rd());                                        // seed the generator
        std::uniform_real_distribution<double> distribution(0.1, 0.5); // define the range

        double beta; // Control parameter of the PIE algorithm

        while (this->R > tolerance && nIter < maximumIterations) // While the trace error is above the tolerance and the maximum number of iterations is not reached
        {
            randomIndexes = this->randomIndexShuffle(); // Get a random array of indexes
            beta = distribution(gen);                   // Get a random beta between 0.1 and 0.5
            for (int i = 0; i < this->N; i++)
            {
                this->computeNextField(randomIndexes[i], beta); // Compute the next value of the field for the given random index
            }

            this->computeAmk(this->result->getSpectrum()); // Compute Amk from the spectrum
            this->computeSmk(this->result->getField());    // Compute Smk from the field
            this->computeSmn();                            // Compute Smn from Smk
            this->computeTmn();                            // Compute Tmn from Smn
            this->computeMu();                             // Compute mu, depends on Tmn
            this->computeResiduals();                      // Compute residuals, depends on mu and Tmn
            this->computeTraceError();                     // Compute trace error, depends on residuals
            this->allTraceErrors.push_back(this->R);       // Store the trace error

            if (this->R < this->bestError) // If the current trace error is the best
            {
                this->bestError = this->R;                  // Update the best error
                this->bestField = this->result->getField(); // Update the best field
            }

            if (this->verbose)
            {
                std::cout << "Iteration = " << nIter + 1 << "\t"
                          << "R = " << this->R << std::endl;
            }

            nIter++;
        }

        std::cout << "Best retrieval error R = " << this->bestError << std::endl;

        this->result->setField(this->bestField); // Set the best field as the result of the retrieval. Updates also the spectrum!

        this->allTraceErrors.push_back(this->bestError); //! The last value of the array is the best result. Not the last retrieval result.

        return *this->result;
    }
};

/**
 * @brief COPRA algorithm for pulse retrieval of SHG-FROG traces.
 *
 */
class COPRA : public retrieverBase
{
private:
    double previousMaxGradient; // Previous maximum gradient
    double currentMaxGradient;  // Current maximum gradient
    double etar;                // Step size for the gradient descent of residuals (global iteration)
    double etaz;                // Step size for the gradient descent of Z (global iteration)

    double alpha = 0.25; //! Should change as an argument in some function. It is the step size for the gradient descent of Z (local iteration)

    std::vector<std::vector<std::complex<double>>> gradrmk; // Gradient of the residuals (global iteration)

    std::vector<std::complex<double>> bestSpectrum; // Result of the best spectrum for retrieval

    /**
     * @brief Computes the next value of the signal operator in the time domain after projection.
     *
     * The projection is given by:
     * S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
     *
     */
    void computeNextSmk()
    {
        std::vector<std::vector<double>> absSmn(this->N, std::vector<double>(this->N));
        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < this->N; ++j)
            {
                absSmn[i][j] = std::abs(this->Smn[i][j]); // |Sₘₙ|
            }
        }

        std::vector<std::complex<double>> nextSmn(this->N);
        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < this->N; ++j)
            {
                if (absSmn[i][j] > 0.0)
                {
                    nextSmn[j] = this->Smn[i][j] / absSmn[i][j] * sqrt(this->Tmeas[i][j] / this->mu); // Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ
                }
                else
                {
                    nextSmn[j] = sqrt(this->Tmeas[i][j] / this->mu);
                }
            }

            this->nextSmk[i] = this->_ft->backwardTransform(nextSmn); // S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
        }
    }

    /**
     * @brief Computes the next value of the signal operator in the time domain after projection.
     *
     * Only computes the next value of the signal operator in the time domain after projection for the given random index.
     *
     * The projection is given by:
     * S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
     *
     */
    void computeNextSmk(int randomIndex)
    {
        std::vector<double> absSmn(this->N);
        for (int j = 0; j < this->N; ++j)
        {
            absSmn[j] = std::abs(this->Smn[randomIndex][j]); // |Sₘₙ|
        }

        std::vector<std::complex<double>> nextSmn(this->N);
        for (int j = 0; j < this->N; ++j)
        {
            if (absSmn[j] > 0.0)
            {
                nextSmn[j] = this->Smn[randomIndex][j] / absSmn[j] * sqrt(this->Tmeas[randomIndex][j] / this->mu); // Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ
            }
            else
            {
                nextSmn[j] = sqrt(this->Tmeas[randomIndex][j] / this->mu);
            }
        }

        this->nextSmk[randomIndex] = this->_ft->backwardTransform(nextSmn); // S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
    }

    /**
     * @brief Computes the gradient of Z for all indexes.
     *
     * This expression is given by:
     *
     *  ∇ₙZₘ = -4 π Δω / Δt · e^{-i τₘ ωₙ} ℱ{ΔSₘₖ·E*ₖ} + ℱ{ΔSₘₖ·A*ₘₖ}
     *
     *  See more detail in expression (S20) of Niels C. Geib paper's supplement on Common Phase Pulse Retrevial Algorithm.
     */
    void computeGradZ()
    {
        std::vector<std::complex<double>> dSmkEk(this->N);                         // ΔSₘₖ·E*ₖ
        std::vector<std::complex<double>> dSmkAmk(this->N);                        // ΔSₘₖ·A*ₘₖ
        std::vector<std::complex<double>> currentField = this->result->getField(); // Eₖ

        for (int i = 0; i < this->N; i++)
        {
            this->gradZ[i] = 0;
        }

        for (int m = 0; m < this->N; m++)
        {
            for (int k = 0; k < this->N; k++)
            {
                dSmkEk[k] = (this->nextSmk[m][k] - this->Smk[m][k]) * std::conj(currentField[k]);  // Sets ΔSₘₖ·E*ₖ
                dSmkAmk[k] = (this->nextSmk[m][k] - this->Smk[m][k]) * std::conj(this->Amk[m][k]); // Sets ΔSₘₖ·A*ₘₖ
            }

            dSmkEk = this->_ft->forwardTransform(dSmkEk);   // ℱ{ΔSₘₖ·E*ₖ}
            dSmkAmk = this->_ft->forwardTransform(dSmkAmk); // ℱ{ΔSₘₖ·A*ₘₖ}

            for (int n = 0; n < this->N; n++)
            {
                this->gradZ[n] += std::conj(this->delays[m][n]) * dSmkEk[n] + dSmkAmk[n]; // e^{-i τₘ ωₙ} ℱ{ΔSₘₖ·E*ₖ} + ℱ{ΔSₘₖ·A*ₘₖ}
            }
        }

        for (int i = 0; i < this->N; i++)
        {
            // Multiply by common factor -4 π Δω / Δt
            this->gradZ[i] *= -4 * M_PI * this->_ft->deltaOmega / (this->_ft->deltaT);
        }
    }

    /**
     * @brief Computes the gradient of Z for the given random index.
     *
     * This expression is given by:
     *
     *  ∇ₙZₘ = -4 π Δω / Δt · e^{-i τₘ ωₙ} ℱ{ΔSₘₖ·E*ₖ} + ℱ{ΔSₘₖ·A*ₘₖ}
     *
     *  Where m is the random index.
     *
     *  See more detail in expression (S20) of Niels C. Geib paper's supplement on Common Phase Pulse Retrevial Algorithm.
     *
     * @param randomIndex Index of the random delay to compute.
     */
    void computeGradZ(int chosenIndex)
    {
        std::vector<std::complex<double>> dSmkEk(this->N);                         // ΔSₘₖ·E*ₖ
        std::vector<std::complex<double>> dSmkAmk(this->N);                        // ΔSₘₖ·A*ₘₖ
        std::vector<std::complex<double>> currentField = this->result->getField(); // Eₖ
        for (int i = 0; i < this->N; ++i)
        {
            dSmkEk[i] = (this->nextSmk[chosenIndex][i] - this->Smk[chosenIndex][i]) * std::conj(currentField[i]);            // Sets ΔSₘₖ·E*ₖ
            dSmkAmk[i] = (this->nextSmk[chosenIndex][i] - this->Smk[chosenIndex][i]) * std::conj(this->Amk[chosenIndex][i]); // Sets ΔSₘₖ·A*ₘₖ
        }

        dSmkEk = this->_ft->forwardTransform(dSmkEk);   // ℱ{ΔSₘₖ·E*ₖ}
        dSmkAmk = this->_ft->forwardTransform(dSmkAmk); // ℱ{ΔSₘₖ·A*ₘₖ}

        for (int i = 0; i < this->N; i++)
        {
            this->gradZ[i] = -4 * M_PI * this->_ft->deltaOmega / (this->_ft->deltaT) * (std::conj(this->delays[chosenIndex][i]) * dSmkEk[i] + dSmkAmk[i]);
        }
    }

    void computeZ()
    {
        this->Z = 0;

        for (int i = 0; i < this->N; i++)
        {
            for (int j = 0; j < this->N; j++)
            {
                this->Z += std::norm(this->nextSmk[i][j] - this->Smk[i][j]);
            }
        }
    }

    void computeZ(int randomIndex)
    {
        this->Z = 0;

        for (int j = 0; j < this->N; j++)
        {
            this->Z += std::norm(this->nextSmk[randomIndex][j] - this->Smk[randomIndex][j]);
        }
    }

    /**
     * @brief Computes the gradient of the residuals for all indexes.
     *
     * This expression is given by:
     *  ∇ₘₖr = -4µ · 2πΔω / Δt ℱ⁻¹[(Tₘₙᵐᵉᵃˢ - μTₘₙ)Sₘₙ]
     *
     */
    void computeGradrmk()
    {
        std::vector<std::complex<double>> difference(this->N); // Tₘₙᵐᵉᵃˢ - μTₘₙ
        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < this->N; ++j)
            {
                difference[j] = -2 * this->mu * this->_ft->deltaT / (M_PI * this->_ft->deltaOmega) * (this->Tmeas[i][j] - this->mu * this->Tmn[i][j]) * this->Smn[i][j];
            }

            this->gradrmk[i] = this->_ft->backwardTransform(difference);
        }
    }

    /**
     * @brief Updates the value of the spectrum by gradient descent.
     *
     * @param step The step of the gradient descent.
     * @param gradient The gradient of the spectrum.
     */
    void computeNextSpectrum(double step, const std::vector<std::complex<double>> &gradient)
    {
        std::vector<std::complex<double>> currentSpectrum = this->result->getSpectrum();
        for (int i = 0; i < this->N; i++)
        {
            currentSpectrum[i] -= step * gradient[i];
        }

        this->result->setSpectrum(currentSpectrum); // Updates also the field!
    }

    /**
     * @brief Local iteration for the COPRA retrieval process.
     *
     *  It is named this way because it is performed for a constant value of m each time.
     *  The first step of the local iteration is to calculate the new value of the candidate pulse signal
     *  operator, S'ₘₖ, by performing a projection onto the set of pulses that satisfy
     *  Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, so the following projection is performed: S'ₘₖ = μ⁻½ ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ},
     *  where the same value of μ is used for all m's.
     *
     *  Then, the value of the pulse spectrum, Ẽ, is updated through a gradient descent. For this,
     *  Zₘ = ∑ₖ |S'ₘₖ - Sₘₖ|² is defined, so the gradient descent in the j-th local iteration
     *  will be Ẽₙ' = Ẽₙ - γₘʲ ∇ₙZₘ; where γₘʲ is a control of the descent step. For traces with little noise,
     *  a good choice is to take γₘʲ = Zₘ / ∑ₙ|∇ₙZₘ|². In the presence of noise, a better choice is to change
     *  the denominator to gₘʲ = max(gₘ₋₁ʲ, ∑ₙ|∇ₙZₘ|²); with g₋₁ʲ = 0. Thus γₘʲ = Zₘ / max(gₘ₋₁ʲ, g_{M-1}ʲ⁻¹)
     *
     *  After updating Ẽ, the same procedure is repeated with the next value of m.
     *  At the end of a complete local iteration step (all m's), the values of R and µ are updated.
     *  If R does not improve in 5 iterations, the 'global iteration' step is started.
     *
     */
    void localIteration(int randomIndex)
    {
        this->computeAmk(this->result->getSpectrum(), randomIndex); // Compute Amk from the spectrum.
        this->computeSmk(this->result->getField(), randomIndex);    // Compute Smk from the field.
        this->computeSmn(randomIndex);                              // Compute Smn from Smk.
        this->computeTmn(randomIndex);                              // Compute Tmn from Smn.

        // Compute projection on Smk
        this->computeNextSmk(randomIndex);

        this->computeZ(randomIndex);     // Compute Z.
        this->computeGradZ(randomIndex); // Compute gradient of Z.

        double gradNorm = this->computeGradZNorm(); // Compute norm of the gradient of Z.
        if (gradNorm > this->currentMaxGradient)
        {
            this->currentMaxGradient = gradNorm;
        }

        // In COPRA, the step size for Z is given by γₘʲ = Zₘ / max(gₘ₋₁ʲ, g_{M-1}ʲ⁻¹)
        this->gamma = this->Z;
        if (this->currentMaxGradient > this->previousMaxGradient)
        {
            this->gamma /= this->currentMaxGradient;
        }
        else
        {
            this->gamma /= this->previousMaxGradient;
        }

        this->computeNextSpectrum(this->gamma, this->gradZ); // Compute next spectrum by gradient descent.
    }

    /**
     * @brief Global iteration for the COPRA retrieval process.
     *
     * The global iteration is performed when the local iteration does not improve the trace error in 5 iterations.
     *
     * Global iteration. In this step, all τₘ are processed at once. It starts with the best solution Ẽ
     * obtained in the local iteration. The first step of the global iteration is to update the values of
     * Sₘₖ, Sₘₙ and Tₘₙ for the candidate pulse, as well as µ and R.
     *
     * Next, we look for a new value of the candidate pulse signal operator S'ₘₖ, by performing a
     * minimization of r with respect to Sₘₖ, through a gradient descent step, S'ₘₖ = Sₘₖ - ηᵣ ∇ₘₖr.
     * Where ηᵣ = α·(r / ∑ₗⱼ |∇ₗⱼr|²), being α a control of the step in each iteration, which we take constant
     * and equal to 0.25 (it has been tested that it gives good convergence). The gradient ∇ₘₖr will be given by the expression:
     * ∇ₘₖr = -4µ · 2πΔω / Δt ℱ⁻¹[(Tₘₙᵐᵉᵃˢ - μTₘₙ)Sₘₙ]
     *
     * Next, we seek to find the corresponding spectrum from the new estimate S'ₘₖ in a step
     * similar to the local iteration, but updating all m's at once. We perform the following
     * gradient descent: Ẽₙ' = Ẽₙ - η_z ∇ₙZ, with η_z = α·(Z / ∑ₖ |∇ₖZ|²).
     *
     */
    void globalIteration()
    {
        this->computeAmk(this->result->getSpectrum()); // Compute Amk from the spectrum
        this->computeSmk(this->result->getField());    // Compute Smk from the field
        this->computeSmn();                            // Compute Smn from Smk
        this->computeTmn();                            // Compute Tmn from Smn
        this->computeMu();                             // Compute mu, depends on Tmn
        this->computeResiduals();                      // Compute residuals, depends on mu and Tmn

        this->computeGradrmk();                                          // Compute gradient of the residuals
        this->etar = this->alpha * this->r / this->computeGradrmkNorm(); // ηᵣ = α·(r / ∑ₗⱼ |∇ₗⱼr|²)

        this->nextSmkGradDescent(); // Compute next Smk by gradient descent

        this->computeZ();     // Compute Z
        this->computeGradZ(); // Compute gradient of Z

        this->etaz = this->alpha * this->Z / this->computeGradZNorm(); // η_z = α·(Z / ∑ₖ |∇ₖZ|²)

        this->computeNextSpectrum(this->etaz, this->gradZ); // Compute next spectrum by gradient descent. Step size is η_z
    }

    /**
     * @brief Computes an array of random indexes between 0 and N-1.
     *
     * @return std::vector<int> the array of random indexes.
     */
    std::vector<int> randomIndexShuffle()
    {
        std::vector<int> indices(this->N);
        std::random_device rd;
        std::mt19937 rng(rd());

        std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., N-1

        // Shuffle the array of indices
        std::shuffle(indices.begin(), indices.end(), rng);

        return indices;
    }

    /**
     * @brief Computes the norm of the gradient of Z.
     *
     * @return double The norm of the gradient of Z.
     */
    double computeGradZNorm()
    {
        double sum = 0;
        for (int i = 0; i < this->N; i++)
        {
            sum += std::norm(this->gradZ[i]);
        }
        return sum;
    }

    /**
     * @brief Computes the norm of the gradient of the residuals.
     *
     * @return double The norm of the gradient of the residuals.
     */
    double computeGradrmkNorm()
    {
        double sum = 0;
        for (int m = 0; m < this->N; m++)
        {
            for (int k = 0; k < this->N; k++)
            {
                sum += std::norm(this->gradrmk[m][k]);
            }
        }
        return sum;
    }

    /**
     * @brief Computes the next value of the signal operator by doing a gradient descent.
     *
     */
    void nextSmkGradDescent()
    {
        for (int m = 0; m < this->N; m++)
        {
            for (int k = 0; k < this->N; k++)
            {
                this->nextSmk[m][k] = this->Smk[m][k] - this->etar * this->gradrmk[m][k];
            }
        }
    }

public:
    /**
     * @brief Construct a new COPRA object
     *
     * @param ft The Fourier transform object to perform fast fourier transforms
     * @param Tmeasured  Measured trace of the pulse to retrieve
     */
    COPRA(FourierTransform &ft, std::vector<std::vector<double>> Tmeasured) : retrieverBase(ft, Tmeasured)
    {
        this->gradrmk.resize(this->N, std::vector<std::complex<double>>(this->N));
        this->bestSpectrum.reserve(this->N);
    }
    /**
     * @brief Construct a new COPRA object
     *
     * @param ft  The Fourier transform object to perform fast fourier transforms
     * @param Tmeasured  Measured trace of the pulse to retrieve
     * @param candidateField  Candidate field to start the retrieval
     */
    COPRA(FourierTransform &ft, std::vector<std::vector<double>> Tmeasured, std::vector<std::complex<double>> &candidateField) : retrieverBase(ft, Tmeasured, candidateField)
        {
        this->gradrmk.resize(this->N, std::vector<std::complex<double>>(this->N));
        this->bestSpectrum.reserve(this->N);
    }
    /**
     * @brief Construct a new COPRA object
     *
     * @param ft  The Fourier transform object to perform fast fourier transforms
     * @param Tmeasured  Measured trace of the pulse to retrieve
     * @param measuredDelays  Delays of the experimentally measured trace
     */
    COPRA(FourierTransform &ft, std::vector<std::vector<double>> Tmeasured, std::vector<double> measuredDelays) : retrieverBase(ft, Tmeasured)
    {
        this->gradrmk.resize(this->N, std::vector<std::complex<double>>(this->N));
        this->bestSpectrum.reserve(this->N);

        // Set up the delays by the given measured delays
        for (int i = 0; i < this->N; i++) // iterates through delay values
        {
            for (int j = 0; j < this->N; j++) // iterates through frequency values
            {
                this->delays[i][j] = std::exp(std::complex<double>(0, this->_ft->omega[j] * measuredDelays[i])); // delay in the time domain by τ
            }
        }
    }

    /**
     * @brief  The COPRA method consists of two main steps:
     *
     *    - Step 1: Local iteration. It is named this way because it is performed for a constant value of m each time.
     *         The first step of the local iteration is to calculate the new value of the candidate pulse signal
     *         operator, S'ₘₖ, by performing a projection onto the set of pulses that satisfy
     *         Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, so the following projection is performed: S'ₘₖ = μ⁻½ ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ},
     *         where the same value of μ is used for all m's.
     *
     *         Then, the value of the pulse spectrum, Ẽ, is updated through a gradient descent. For this,
     *         Zₘ = ∑ₖ |S'ₘₖ - Sₘₖ|² is defined, so the gradient descent in the j-th local iteration
     *         will be Ẽₙ' = Ẽₙ - γₘʲ ∇ₙZₘ; where γₘʲ is a control of the descent step. For traces with little noise,
     *         a good choice is to take γₘʲ = Zₘ / ∑ₙ|∇ₙZₘ|². In the presence of noise, a better choice is to change
     *         the denominator to gₘʲ = max(gₘ₋₁ʲ, ∑ₙ|∇ₙZₘ|²); with g₋₁ʲ = 0. Thus γₘʲ = Zₘ / max(gₘ₋₁ʲ, g_{M-1}ʲ⁻¹)
     *
     *         After updating Ẽ, the same procedure is repeated with the next value of m.
     *         At the end of a complete local iteration step (all m's), the values of R and µ are updated.
     *         If R does not improve in 5 iterations, the 'global iteration' step is started.
     *
     *      - Step 2: Global iteration. In this step, all τₘ are processed at once. It starts with the best solution Ẽ
     *         obtained in the local iteration. The first step of the global iteration is to update the values of
     *         Sₘₖ, Sₘₙ and Tₘₙ for the candidate pulse, as well as µ and R.
     *
     *         Next, we look for a new value of the candidate pulse signal operator S'ₘₖ, by performing a
     *         minimization of r with respect to Sₘₖ, through a gradient descent step, S'ₘₖ = Sₘₖ - ηᵣ ∇ₘₖr.
     *         Where ηᵣ = α·(r / ∑ₗⱼ |∇ₗⱼr|²), being α a control of the step in each iteration, which we take constant
     *         and equal to 0.25 (it has been tested that it gives good convergence). The gradient ∇ₘₖr will be given by the expression:
     *         ∇ₘₖr = -4µ · 2πΔω / Δt ℱ⁻¹[(Tₘₙᵐᵉᵃˢ - μTₘₙ)Sₘₙ]
     *
     *         Next, we seek to find the corresponding spectrum from the new estimate S'ₘₖ in a step
     *         similar to the local iteration, but updating all m's at once. We perform the following
     *         gradient descent: Ẽₙ' = Ẽₙ - η_z ∇ₙZ, with η_z = α·(Z / ∑ₖ |∇ₖZ|²).
     *
     *         After this, we can calculate the trace error R, and if the convergence condition is satisfied or the maximum number of iterations is reached, the algorithm stops.
     *         Otherwise, return to the local iteration step.
     *
     * @param tolerance Tolerance of the retrieval. The retrieval will stop when the trace error is below this value.
     * @param maximumIterations Maximum number of iterations for the retrieval.
     * @return Pulse A Pulse class instance containing the resulting pulse of the retrieval.
     */
    Pulse retrieve(double tolerance, double maximumIterations)
    {
        int nIter = 0;                     // Number of iterations
        int stepsSinceLastImprovement = 0; // Number of steps since the last improvement of the result
        bool mode = 1;                     // 1 for local iteration, 0 for global iteration
        std::vector<int> randomIndexes;    // Array of random indexes

        this->bestError = std::numeric_limits<double>::infinity(); // Best error is set to infinity

        this->computeAmk(this->result->getSpectrum()); // Compute Amk from the spectrum
        this->computeSmk(this->result->getField());    // Compute Smk from the field
        this->computeSmn();                            // Compute Smn from Smk
        this->computeTmn();                            // Compute Tmn from Smn
        this->computeMu();                             // Compute mu, depends on Tmn
        this->computeResiduals();                      // Compute residuals, depends on mu and Tmn
        this->computeTraceError();                     // Compute trace error, depends on residuals
        this->allTraceErrors.push_back(this->R);       // Store the first value of the trace error

        this->computeNextSmk();       // Compute projection on Smk
        this->currentMaxGradient = 0; // Current maximum gradient is set to 0
        // Let's compute the current max gradient
        for (int m = 0; m < this->N; m++)
        {
            this->computeGradZ(m);

            this->previousMaxGradient = this->computeGradZNorm();
            if (this->previousMaxGradient > this->currentMaxGradient)
            {
                this->currentMaxGradient = this->previousMaxGradient;
            }
        }

        // Main loop. While the trace error is above the tolerance and the maximum number of iterations is not reached
        while (this->R > tolerance && nIter < maximumIterations)
        {

            if (mode) // If we are in local iteration
            {
                this->previousMaxGradient = this->currentMaxGradient; // Previous maximum gradient is the current maximum gradient
                this->currentMaxGradient = 0;                         // Current maximum gradient is set to 0

                randomIndexes = this->randomIndexShuffle(); // Get a random array of indexes
                for (int i = 0; i < this->N; i++)
                {
                    this->localIteration(randomIndexes[i]); // Compute the next value of the field for the given random index
                }

                this->computeMu();                       // Compute mu, depends on Tmn
                this->computeResiduals();                // Compute residuals, depends on mu and Tmn
                this->computeTraceError();               // This trace error is with the approximation, as the spectrum changed every iteration
                this->allTraceErrors.push_back(this->R); // Store the trace error. !! This error is approximated, as the spectrum changed every iteration but is not processed for all spectrums

                if (this->R >= this->bestError)
                {
                    stepsSinceLastImprovement++;        // If the result is not improved, increase the number of steps since the last improvement
                    if (stepsSinceLastImprovement == 5) // If the result is not improved in 5 steps, start global iteration
                    {
                        mode = 0; // Change to global iteration
                        if (this->verbose)
                        {
                            std::cout << "Local iteration ended, starting global iteration" << std::endl;
                        }
                        // We pick the best result from the local iteration to start the global iteration
                        this->result->setSpectrum(this->bestSpectrum); // Updates also the field!
                    }
                }
                else // If the result is improved
                {
                    stepsSinceLastImprovement = 0;                    // Reset the number of steps since the last improvement
                    this->bestError = this->R;                        // Update the best error. !! Approximated value, as the spectrum changed every iteration but is not processed for all spectrums
                    this->bestSpectrum = this->result->getSpectrum(); // Update the best spectrum. Also updates the field!
                }
            }
            else // If we are in global iteration
            {
                this->globalIteration(); // Perform global iteration

                this->computeAmk(this->result->getSpectrum()); // Compute Amk from the spectrum
                this->computeSmk(this->result->getField());    // Compute Smk from the field
                this->computeSmn();                            // Compute Smn from Smk
                this->computeTmn();                            // Compute Tmn from Smn
                this->computeMu();                             // Compute mu, depends on Tmn
                this->computeResiduals();                      // Compute residuals, depends on mu and Tmn
                this->computeTraceError();                     // Compute trace error, depends on residuals
                this->allTraceErrors.push_back(this->R);       // Store the trace error

                if (this->R < this->bestError) // If the current trace error is the best
                {
                    this->bestError = this->R;                        // Update the best error
                    this->bestSpectrum = this->result->getSpectrum(); // Update the best spectrum
                }
            }

            if (this->verbose)
            {
                std::cout << "Iteration = " << nIter + 1 << "\t"
                          << "R = " << this->R << std::endl;
            }

            nIter++; // Increase the number of iterations
        }

        this->result->setSpectrum(this->bestSpectrum); // Set the best spectrum as the result of the retrieval. Updates also the field!
        if (mode)
        {                                                  // If the result was achieved in local iteration, compute R as it was shown as an approximated value.
            this->computeAmk(this->result->getSpectrum()); // Compute Amk from the spectrum
            this->computeSmk(this->result->getField());    // Compute Smk from the field
            this->computeSmn();                            // Compute Smn from Smk
            this->computeTmn();                            // Compute Tmn from Smn
            this->computeMu();                             // Compute mu, depends on Tmn
            this->computeResiduals();                      // Compute residuals, depends on mu and Tmn
            this->computeTraceError();                     // Compute trace error, depends on residuals
            this->bestError = this->R;                     // Update the best error
            std::cout << "Best retrieval error found in local iteration" << std::endl;
        }

        std::cout << "Best retrieval error R = " << this->bestError << std::endl;

        this->allTraceErrors.push_back(this->bestError); //! Last stored value is the best result, not the last computed error!

        return *this->result;
    }
};

#endif // RETRIEVERS_INCLUDED