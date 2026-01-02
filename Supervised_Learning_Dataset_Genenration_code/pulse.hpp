/**
 * @file pulse.hpp
 * @author Víctor Loras Herrero
 * @brief Provides a class to simulate an ultrashort optical pulse using its envelope description.
 *
 * @copyright Copyright (c) 2023
 *
 * Check out Nils C Geib "PyPret" Pulse module which inspired this code.
 * https://pypret.readthedocs.io/en/latest/apidoc/pypret.pulse.html
 *
 */

#ifndef PULSE_INCLUDED
#define PULSE_INCLUDED
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <complex>
#include <vector>
#include <random>
#include <functional>
#include "fourier.hpp"
#include "utils.hpp"

/**
 * @brief Brent's method to find the root of a mathematical function.
 *
 * Brent's method combines root bracketing, interval bisection, and inverse quadratic interpolation.
 * It has the reliability of bisection but it can be as quick as some less-reliable methods when the function's root is smooth.
 *
 * This code has been taken from the following StackOverflow answer:
 * https://codereview.stackexchange.com/questions/103762/implementation-of-brents-algorithm-to-find-roots-of-a-polynomial
 *
 * @param f The function for which we are trying to approximate a solution f(x)=0.
 * @param lower_bound The lower boundary for the range in which to search for a solution.
 * @param upper_bound The upper boundary for the range in which to search for a solution.
 * @param TOL The tolerance which the function uses to determine if it has found the root of the function.
 * @param MAX_ITER The maximum number of iterations to perform before the function stops.
 *
 * @return void
 */
void brents_fun(std::function<double(double)> f, double lower_bound, double upper_bound, const double TOL, const double MAX_ITER)
{
    double a = lower_bound;
    double b = upper_bound;
    double fa = f(a); // calculated now to save function calls
    double fb = f(b); // calculated now to save function calls
    double fs = 0;    // initialize

    if (!(fa * fb < 0))
    {
        std::cout << "Signs of f(lower_bound) and f(upper_bound) must be opposites" << std::endl; // throws exception if root isn't bracketed
        return;
    }

    if (std::abs(fa) < std::abs(b)) // if magnitude of f(lower_bound) is less than magnitude of f(upper_bound)
    {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;      // c now equals the largest magnitude of the lower and upper bounds
    double fc = fa;    // precompute function evalutation for point c by assigning it the same value as fa
    bool mflag = true; // boolean flag used to evaluate if statement later on
    double s = 0;      // Our Root that will be returned
    double d = 0;      // Only used if mflag is unset (mflag == false)

    for (unsigned int iter = 1; iter < MAX_ITER; ++iter)
    {
        // stop if converged on root or error is less than tolerance
        if (std::abs(b - a) < TOL)
        {
            // std::cout << "After " << iter << " iterations the root is: " << s << std::endl;
            return;
        } // end if

        if (fa != fc && fb != fc)
        {
            // use inverse quadratic interopolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))) + (b * fa * fc / ((fb - fa) * (fb - fc))) + (c * fa * fb / ((fc - fa) * (fc - fb)));
        }
        else
        {
            // secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        /*
            Crazy condition statement!:
            -------------------------------------------------------
            (condition 1) s is not between  (3a+b)/4  and b or
            (condition 2) (mflag is true and |s−b| ≥ |b−c|/2) or
            (condition 3) (mflag is false and |s−b| ≥ |c−d|/2) or
            (condition 4) (mflag is set and |b−c| < |TOL|) or
            (condition 5) (mflag is false and |c−d| < |TOL|)
        */
        if (((s < (3 * a + b) * 0.25) || (s > b)) ||
            (mflag && (std::abs(s - b) >= (std::abs(b - c) * 0.5))) ||
            (!mflag && (std::abs(s - b) >= (std::abs(c - d) * 0.5))) ||
            (mflag && (std::abs(b - c) < TOL)) ||
            (!mflag && (std::abs(c - d) < TOL)))
        {
            // bisection method
            s = (a + b) * 0.5;

            mflag = true;
        }
        else
        {
            mflag = false;
        }

        fs = f(s); // calculate fs
        d = c;     // first time d is being used (wasnt used on first iteration because mflag was set)
        c = b;     // set c equal to upper bound
        fc = fb;   // set f(c) = f(b)

        if (fa * fs < 0) // fa and fs have opposite signs
        {
            b = s;
            fb = fs; // set f(b) = f(s)
        }
        else
        {
            a = s;
            fa = fs; // set f(a) = f(s)
        }

        if (std::abs(fa) < std::abs(fb)) // if magnitude of fa is less than magnitude of fb
        {
            std::swap(a, b);   // swap a and b
            std::swap(fa, fb); // make sure f(a) and f(b) are correct after swap
        }

    } // end for

    std::cout << "The solution does not converge or iterations are not sufficient" << std::endl;
} // end brent_fun

/**
 * @class Pulse
 * @brief Represents a pulse in the time and frequency domain.
 *
 * This class provides methods for manipulating and analyzing pulses.
 * It uses Fourier transforms to switch between the time and frequency domain.
 */
class Pulse
{
private:
    FourierTransform *_ft; ///< Fourier transform object used for time-frequency conversions.

    std::vector<std::complex<double>> _field;    ///< The pulse in the time domain.
    std::vector<std::complex<double>> _spectrum; ///< The pulse in the frequency domain.

    // Random pulse generation variables
    double _tbp;          ///< Time-bandwidth product.
    double t0;            ///< Central time of the pulse.
    double omega0;        ///< Central frequency of the pulse.
    double temporalWidth; ///< Temporal width of the pulse.

    std::vector<std::complex<double>> candidateField; ///< Candidate field for pulse shaping. This is used in the randomPulse method.

    /**
     * @brief Objective function for pulse shaping.
     *
     * Used in the randomPulse method.
     * This function calculates the difference between the desired time-bandwidth product and the time-bandwidth product of the candidate field after applying a temporal filter.
     *
     * @param factor The factor by which to scale the temporal width of the Gaussian filter.
     * @return The difference between the desired and actual time-bandwidth product.
     */
    double objective(double const &factor)
    {

        std::vector<double> temporalFilter = gaussian(this->_ft->t, this->t0, this->temporalWidth * factor); // Gaussian filter in the time domain

        std::vector<std::complex<double>> result(this->N); // E(t) * Gaussian(t)

        for (int i = 0; i < this->N; i++)
        {
            result[i] = this->candidateField[i] * temporalFilter[i]; // E(t) * Gaussian(t)
        }

        this->setField(result); // Set the field to the filtered candidate field

        return this->_tbp - this->getTimeBandwidthProduct(); // Return the difference between the desired and actual time-bandwidth product
    }

public:
    int N; ///< Number of points in the time and frequency domain.

    /**
     * @brief Construct a new Pulse object.
     *
     * @param ft Fourier transform object used for time-frequency conversions.
     */
    Pulse(FourierTransform &ft)
    {
        this->_ft = &ft;
        this->N = this->_ft->N;
    }

    /**
     * @brief Sets the pulse in the time domain.
     *
     * Stores the value in the _field attribute and updates the _spectrum attribute.
     *
     * @param val The pulse in the time domain.
     *
     */
    void setField(const std::vector<std::complex<double>> &val)
    {
        this->_field = val;
        this->updateSpectrum();
    }

    /**
     * @brief Sets the pulse in the frequency domain.
     *
     * Stores the value in the _spectrum attribute and updates the _field attribute.
     *
     * @param val The pulse in the frequency domain.
     *
     */
    void setSpectrum(const std::vector<std::complex<double>> &val)
    {
        this->_spectrum = val;
        this->updateField();
    }

    /**
     * @brief Updates the pulse in the time domain.
     *
     * Updates the _field attribute using the _spectrum attribute.
     *
     */
    void updateField()
    {
        this->_field = this->_ft->backwardTransform(this->_spectrum);
    }

    /**
     * @brief Updates the pulse in the frequency domain.
     *
     * Updates the _spectrum attribute using the _field attribute.
     *
     */
    void updateSpectrum()
    {
        this->_spectrum = this->_ft->forwardTransform(this->_field);
    }

    /**
     * @brief Returns the pulse in the time domain.
     *
     * @return The pulse in the time domain.
     */
    std::vector<std::complex<double>> getField()
    {
        return this->_field;
    }

    /**
     * @brief Returns the pulse in the frequency domain.
     *
     * @return The pulse in the frequency domain.
     */
    std::vector<std::complex<double>> getSpectrum()
    {
        return this->_spectrum;
    }

    /**
     * @brief Returns the intensity of the pulse in the time domain.
     *
     * @return The intensity of the pulse in the time domain as a vector of doubles.
     */
    std::vector<double> getIntensity()
    {
        std::vector<double> intensity(this->N);

        for (int i = 0; i < this->N; ++i)
        {
            intensity[i] = std::norm(this->_field[i]);
        }

        return intensity;
    }

    /**
     * @brief Returns the amplitude of the pulse in the time domain.
     *
     * @return The amplitude of the pulse in the time domain as a vector of doubles.
     */
    std::vector<double> getAmplitude()
    {
        std::vector<double> amplitude(this->N);

        for (int i = 0; i < this->N; ++i)
        {
            amplitude[i] = std::abs(this->_field[i]);
        }

        return amplitude;
    }

    /**
     * @brief Returns the phase of the pulse in the time domain.
     *
     * @return The phase of the pulse in the time domain as a vector of doubles.
     */
    std::vector<double> getPhase()
    {
        return unwrapPhase(this->_field);
    }

    /**
     * @brief Returns the intensity of the pulse in the frequency domain.
     *
     * @return The intensity of the pulse in the frequency domain as a vector of doubles.
     */
    std::vector<double> getSpectralIntensity()
    {
        std::vector<double> spectralIntensity(this->N);

        for (int i = 0; i < this->N; i++)
        {
            spectralIntensity[i] = std::norm(this->_spectrum[i]);
        }

        return spectralIntensity;
    }

    /**
     * @brief Returns the amplitude of the pulse in the frequency domain.
     *
     * @return The amplitude of the pulse in the frequency domain as a vector of doubles.
     */
    std::vector<double> getSpectralAmplitude()
    {
        std::vector<double> spectralAmplitude(this->N);

        for (int i = 0; i < this->N; ++i)
        {
            spectralAmplitude[i] = std::abs(this->_spectrum[i]);
        }

        return spectralAmplitude;
    }

    /**
     * @brief Returns the phase of the pulse in the frequency domain.
     *
     * @return The phase of the pulse in the frequency domain as a vector of doubles.
     */
    std::vector<double> getSpectralPhase()
    {
        return unwrapPhase(this->_spectrum);
    }

    /**
     * @brief Returns the time-bandwidth product of the pulse.
     *
     * Time-bandwidth product is defined as the product of the pulse duration and the spectral width,
     * TBP = Δt * Δω,
     * It is computed as the standard deviation of the pulse in the time domain multiplied by the standard deviation of the pulse in the frequency domain.
     *
     * @return The time-bandwidth product of the pulse.
     */
    double getTimeBandwidthProduct()
    {
        return stdDev(this->_ft->t, this->getIntensity()) * stdDev(this->_ft->omega, this->getSpectralIntensity());
    }

    /**
     * @brief Returns the SHG-FROG trace of the pulse.
     *
     * The SHG-FROG trace is defined as the following expression:
     * T(ω, τ) =  | ∫ E(t)E(t - τ) exp(-i ω t) dt |² = |FT[E(t)E(t - τ)]|²
     *
     * @return The SHG-FROG trace of the pulse. It is a N x N matrix of doubles.
     */
    std::vector<std::vector<double>> getTrace()
    {
        std::vector<double> tau(this->N); // τ, delays
        for (int i = 0; i < this->N; i++)
        {
            tau[i] = (i - std::floor(0.5 * this->N)) * this->_ft->deltaT; // τ = (i - N/2) * Δt
        }

        // The delay can be introduced in the frequency domain by multiplying the spectrum by exp(-i ω τ)
        // This is convenient to avoid the need to interpolate the field in the time domain
        // Also simplifies the calculation of the trace and other terms in the retrieval algorithms
        std::vector<std::vector<std::complex<double>>> delayed_spectrum(this->N, std::vector<std::complex<double>>(this->N));

        for (int i = 0; i < this->N; i++) // iterates through delay values
        {
            for (int j = 0; j < this->N; j++) // iterates through frequency values
            {
                delayed_spectrum[i][j] = this->_spectrum[j] * std::exp(std::complex<double>(0, this->_ft->omega[j] * tau[i])); // delay in the time domain by τ
            }
        }

        std::vector<std::complex<double>> delayed_pulse(this->N);                             // E(t - τ)
        std::vector<std::complex<double>> signal_operator(this->N);                           // E(t)E(t - τ)
        std::vector<std::vector<double>> trace_values(this->N, std::vector<double>(this->N)); // T(ω, τ)

        for (int i = 0; i < N; i++)
        {
            delayed_pulse = this->_ft->backwardTransform(delayed_spectrum[i]); // E(t - τ)
            for (int j = 0; j < N; j++)
            {
                signal_operator[j] = this->_field[j] * delayed_pulse[j]; // E(t)E(t - τ)
            }

            signal_operator = this->_ft->forwardTransform(signal_operator); // FT{E(t)E(t - τ)}

            for (int j = 0; j < N; j++)
            {
                trace_values[i][j] = std::norm(signal_operator[j]); // |FT{E(t)E(t - τ)}|²
            }
        }

        return trace_values;
    }

    /**
     * @brief Creates a random pulse with a specified time-bandwidth product.
     *
     * This method uses the algorithm described in Nils C Geib "PyPret" Pulse module.
     * https://pypret.readthedocs.io/en/latest/apidoc/pypret.pulse.html
     *
     * The algorithm starts from random complex values in the frequency domain, which decay in the extremes of the grid
     * to the float roundoff error. These random complex values are filtered in the frequency domain by a Gaussian filter.
     * It then transforms into the time domain and applies a Gaussian filter,
     * then transforms into the time domain and applies another Gaussian filter. The filter functions are
     * Gaussians with the specified time-bandwidth product.
     * The chosen filter functions only roughly give the correct TBP.
     * To obtain the exact result we scale the temporal filter bandwidth by a factor and perform a scalar minimization on that value.
     *
     * The larger the time-bandwidth product, the more points are needed in the time and frequency domain.
     *
     * @param TBP The desired time-bandwidth product.
     *
     * @return 1 if the pulse was created successfully, 0 otherwise.
     */
    bool randomPulse(double TBP)
    {
        this->_tbp = TBP; // Store the desired time-bandwidth product

        this->t0 = 0.5 * (this->_ft->t[0] + this->_ft->t[this->N - 1]);             // Central time of the grid
        this->omega0 = 0.5 * (this->_ft->omega[0] + this->_ft->omega[this->N - 1]); // Central frequency of the grid

        // Initialize random number generator
        std::random_device rd;                                 // obtain a random number from hardware
        std::mt19937 gen(rd());                                // seed the generator
        std::uniform_real_distribution<double> dist(0.0, 1.0); // define the range

        // This is roughly the log of the roundoff error induced by an FFT
        double logEdge = std::log(this->N * std::numeric_limits<double>::epsilon());

        // Calculate the width of a Gaussian function that drops exactly to edge_value at the edges of the grid
        double spectralWidth = sqrt(-0.125 * (this->_ft->omega[0] - this->_ft->omega[N - 1]) * (this->_ft->omega[0] - this->_ft->omega[N - 1]) / logEdge);
        // Now the same in the temporal domain
        double maxTemporalWidth = sqrt(-0.125 * (this->_ft->t[0] - this->_ft->t[N - 1]) * (this->_ft->t[0] - this->_ft->t[N - 1]) / logEdge);
        // The actual temporal width is obtained by the uncertainty relation from the specified TBP
        this->temporalWidth = 2.0 * TBP / spectralWidth;

        if (this->temporalWidth > maxTemporalWidth)
        {
            throw std::runtime_error("The required time-bandwidth product cannot be reached. Increase sample number.\n");
            return 0;
        }

        // Special case for TBP = 0.5 (transform-limited case)
        if (TBP == 0.5)
        {

            for (int i = 0; i < this->N; ++i)
            {
                this->_spectrum.push_back(std::exp(std::complex<double>(-0.5 * (this->_ft->omega[i] - omega0) * (this->_ft->omega[i] - omega0) / (spectralWidth * spectralWidth), 2 * M_PI * dist(gen))));
            }

            this->updateField();

            return 1;
        }

        std::vector<double> spectralFilter = gaussian(this->_ft->omega, this->omega0, spectralWidth); // Gaussian filter in the frequency domain

        /*
            The algorithm works by iteratively filtering in the frequency and time
            domain. However, the chosen filter functions only roughly give
            the correct TBP. To obtain the exact result we scale the temporal
            filter bandwidth by a factor and perform a scalar minimization on
            that value.
        */

        // Rough guess for the relative range in which our optimal value lies
        double factorMin = 0.5;
        double factorMax = 1.5;

        std::vector<std::complex<double>> candidateSpectrum(this->N); // Candidate spectrum for pulse shaping

        for (int i = 0; i < N; ++i)
        {
            candidateSpectrum[i] = spectralFilter[i] * dist(gen) * std::exp(std::complex<double>(0.0, 2 * M_PI * dist(gen))); // Random complex values in the frequency domain multiplied by the Gaussian filter
        }

        this->candidateField = this->_ft->backwardTransform(candidateSpectrum); // Transform to the time domain

        // Ensure the objective function changes sign in the chosen bounds
        int iters = 0;
        while (std::signbit(objective(factorMin)) == std::signbit(objective(factorMax)))
        {
            // For some random arrays, this condition is not always fulfilled. Try again.
            for (int i = 0; i < N; ++i)
            {
                candidateSpectrum[i] = spectralFilter[i] * dist(gen) * std::exp(std::complex<double>(0.0, 2 * M_PI * dist(gen))); // New random complex values in the frequency domain multiplied by the Gaussian filter
            }

            this->candidateField = this->_ft->backwardTransform(candidateSpectrum);

            iters++;

            if (iters == 100) // If it fails 100 times, it is very likely that it will not work
            {
                throw std::runtime_error("Could not create a pulse for these parameters!");
                return 0;
            }
        }

        // Create a callable object (lambda) that wraps objective function
        auto objectiveFunction = [&](double factor)
        {
            return this->objective(factor);
        };

        // Perform scalar minimization to find the optimal value for the temporal filter bandwidth
        brents_fun(objectiveFunction, factorMin, factorMax, 1e-12, 1000);
        // The brents_fun already has changed the values of the field and spectrum attributes
        // So there is no need to update them

        return 1;
    }
};

#endif // PULSE_INCLUDED