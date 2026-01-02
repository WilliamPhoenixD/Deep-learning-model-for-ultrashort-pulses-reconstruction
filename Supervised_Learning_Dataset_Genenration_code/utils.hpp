/**
 * @file utils.hpp
 * @author Víctor Loras Herrero
 * @brief Various auxiliary generic functions used in other modules.
 *
 */

#ifndef UTILS_INCLUDED
#define UTILS_INCLUDED
#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include <random>
#include "fourier.hpp"

/**
 * \brief Computes the mean value of the element-wise division of two vectors.
 *
 * This function computes the mean value of the element-wise division of two vectors,
 * `x` and `y`. It calculates the sum of the element-wise product of `x` and `y`,
 * and divides it by the sum of the elements in `y`.
 *
 * \tparam T The type of elements in the vectors.
 * \param x The first vector.
 * \param y The second vector.
 * \return The mean value of the element-wise division of `x` and `y`.
 */
double mean(const std::vector<double> &x, const std::vector<double> &y)
{
    int N = x.size();

    double sumProduct = 0;
    double sumY = 0;

    for (int i = 0; i < N; i++)
    {
        sumProduct += x[i] * y[i];
        sumY += y[i];
    }

    return sumProduct / sumY;
}

/**
 * \brief Computes the standard deviation of a dataset.
 *
 * This function calculates the standard deviation of a dataset represented by two vectors,
 * `x` and `y`. It uses the mean value of `x` and `y` to calculate the standard deviation.
 *
 * \tparam T The type of elements in the vectors.
 * \param x The vector representing the x-values of the dataset.
 * \param y The vector representing the y-values of the dataset.
 * \return The standard deviation of the dataset.
 */
template <typename T>
T stdDev(const std::vector<T> &x, const std::vector<T> &y)
{
    T meanValue = mean(x, y);

    T sumNum = 0;
    T sumY = 0;

    for (size_t i = 0; i < x.size(); i++)
    {
        T diff = x[i] - meanValue;
        sumNum += diff * diff * y[i];
        sumY += y[i];
    }

    return std::sqrt(sumNum / sumY);
}

/**
 * \brief Finds the maximum element in a vector.
 *
 * This function finds and returns the maximum element in the given vector.
 *
 * \tparam T The type of elements in the vector.
 * \param vec The vector to search for the maximum element.
 * \return The maximum element in the vector.
 */
template <typename T>
T findMax(const std::vector<T> &vec)
{
    return *std::max_element(vec.begin(), vec.end());
}

/**
 * \brief Finds the index where the value in the vector exceeds a specified threshold starting from the beggining of the vector.
 *
 * This function searches the given vector for the leftmost index where the value exceeds
 * the specified threshold.
 *
 * \tparam T The type of elements in the vector.
 * \param x The vector to search for the left index.
 * \param value The threshold value to compare with.
 * \return The left index where the value exceeds the threshold, or -1 if not found.
 */
template <typename T>
int leftIndexValue(const std::vector<T> &x, double value)
{
    int N = x.size();

    for (int i = 0; i < N; i++)
    {
        if (x[i] - value > 0)
        {
            return i;
        }
    }

    return -1;
}

/**
 * \brief Finds the first index where the value in the vector exceeds a specified threshold starting from the end of the vector.
 *
 * This function searches the given vector for the rightmost index where the value exceeds
 * the specified threshold.
 *
 * \tparam T The type of elements in the vector.
 * \param x The vector to search for the right index.
 * \param value The threshold value to compare with.
 * \return The right index where the value exceeds the threshold, or -1 if not found.
 */
template <typename T>
int rightIndexValue(const std::vector<T> &x, double value)
{
    int N = x.size();

    for (int i = 0; i < N; i++)
    {
        if (x[N - i - 1] - value > 0)
        {
            return N - i - 1;
        }
    }

    return -1;
}

/**
 * \brief Calculates the Full Width at Half Maximum (FWHM) of a waveform.
 *
 * This function calculates the Full Width at Half Maximum (FWHM) of a waveform
 * given a vector of values. The FWHM is the width of the waveform at half of its maximum value.
 *
 * \tparam T The type of elements in the waveform vector.
 * \tparam Numeric The type of the delta_t parameter.
 * \param x The vector of waveform values.
 * \param delta_t The time interval between waveform samples.
 * \return The FWHM of the waveform.
 */
template <typename T, typename Numeric>
T FWHM(const std::vector<T> &x, Numeric delta_t)
{
    T halfMax = findMax(x) / 2;

    T leftIndex = leftIndexValue(x, halfMax);
    T rightIndex = rightIndexValue(x, halfMax);

    return (rightIndex - leftIndex) * delta_t;
}

/**
 * @brief Calculates a Gaussian function with center `x0` and standard deviation `sigma` for each element in the input vector `x`.
 *
 * @param x The input vector containing the values at which to calculate the Gaussian function.
 * @param x0 The center of the Gaussian function. Default is 0.0.
 * @param sigma The standard deviation of the Gaussian function. Default is 1.0.
 * @return A vector containing the Gaussian values corresponding to each element of the input vector `x`.
 */
std::vector<double> gaussian(const std::vector<double> &x, double x0 = 0.0, double sigma = 1.0)
{
    std::vector<double> result;
    result.reserve(x.size());

    double d;

    for (const auto &element : x)
    {
        d = (element - x0) / sigma;
        result.push_back(std::exp(-0.5 * d * d));
    }

    return result;
}

/**
 * @brief Calculate the unwrapped phase of a vector of complex numbers.
 *
 * This function takes a vector of complex numbers and returns a vector of the
 * unwrapped phase values. The phase values are unwrapped to ensure smooth
 * transitions when the phase crosses the -π to π boundary.
 *
 * @param complexVector A vector of complex numbers for which the phase needs
 *                      to be unwrapped.
 * @return A vector of unwrapped phase values corresponding to the input
 *         complex numbers.
 */
std::vector<double> unwrapPhase(const std::vector<std::complex<double>> &complexVector)
{
    std::vector<double> unwrappedPhase;
    unwrappedPhase.reserve(complexVector.size());

    // Calculate the phase of each complex number
    for (const std::complex<double> &z : complexVector)
    {
        unwrappedPhase.push_back(std::arg(z));
    }

    // Unwrap the phase to ensure smooth transitions across -π to π
    for (size_t i = 1; i < unwrappedPhase.size(); ++i)
    {
        double diff = unwrappedPhase[i] - unwrappedPhase[i - 1];
        if (diff < -M_PI)
        {
            unwrappedPhase[i] += 2 * M_PI;
        }
        else if (diff > M_PI)
        {
            unwrappedPhase[i] -= 2 * M_PI;
        }
    }

    return unwrappedPhase;
}

/**
 * @brief Computes the trace of a pulse given by T(ω, τ) = | ∫ E(t)E(t - τ) exp(-i ω t) dt |²
 *
 * @param x The input vector containing the values of the electric field of the pulse.
 * @param deltaT The time step.
 * @return A 2D vector (NxN) containing the values of the trace.
 */
std::vector<std::vector<double>> trace(const std::vector<std::complex<double>> &x, const std::vector<double> &t, double deltaT)
{
    int N = x.size(); // number of samples

    FourierTransform ft(N, deltaT, t[0]);                                // delays will be introduced as the spectrum multiplied by a phase factor
    std::vector<std::complex<double>> spectrum = ft.forwardTransform(x); // spectrum of the given electric field

    std::vector<double> omega = toAngularFrequency(fftFreq(N, deltaT)); // angular frequencies of the measurement

    std::vector<double> tau(N); // delays

    for (int i = 0; i < N; i++)
    {
        tau[i] = (i - std::floor(0.5 * N)) * deltaT;
    }

    std::vector<std::vector<std::complex<double>>> delayed_spectrum(N, std::vector<std::complex<double>>(N));

    for (int i = 0; i < N; i++) // iterates through delay values
    {
        for (int j = 0; j < N; j++) // iterates through frequency values
        {
            delayed_spectrum[i][j] = spectrum[j] * std::exp(std::complex<double>(0, omega[j] * tau[i])); // delay in the time domain by τ
        }
    }

    std::vector<std::complex<double>> delayed_pulse(N);                       // E(t - τ)
    std::vector<std::complex<double>> signal_operator(N);                     // E(t)E(t - τ)
    std::vector<std::vector<double>> trace_values(N, std::vector<double>(N)); // T(ω, τ)

    for (int i = 0; i < N; i++)
    {
        delayed_pulse = ft.backwardTransform(delayed_spectrum[i]);
        for (int j = 0; j < N; j++)
        {
            signal_operator[j] = x[j] * delayed_pulse[j]; // E(t)E(t - τ)
        }

        signal_operator = ft.forwardTransform(signal_operator); // FT{E(t)E(t - τ)}

        for (int j = 0; j < N; j++)
        {
            trace_values[i][j] = std::norm(signal_operator[j]); // |FT{E(t)E(t - τ)}|²
        }
    }

    return trace_values;
}

/**
 * @brief Adds gaussian noise to a trace.
 *
 * @param originalTrace 2D vector (NxN) containing the values of the trace without noise.
 * @param N Number of samples of the trace's pulse.
 * @param noiseLevel Level of gaussian noise applied to the trace (% of the maximum value).
 * @return 2D vector (NxN) containing the values of the trace with noise.
 */
std::vector<std::vector<double>> add_noise(const std::vector<std::vector<double>> &originalTrace, int N, double noiseLevel)
{
    std::vector<std::vector<double>> noisyTrace(N, std::vector<double>(N));

    // Seed for random number generation
    std::random_device rd;
    std::mt19937 gen(rd());

    // Calculate standard deviation based on the maximum value
    double TmeasMax = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (originalTrace[i][j] > TmeasMax)
            {
                TmeasMax = originalTrace[i][j];
            }
        }
    }

    double stdDev = noiseLevel * TmeasMax;

    // Distribution for random noise
    std::normal_distribution<> distribution(0, 1);

    // Add noise to each element in the matrix
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            noisyTrace[i][j] = originalTrace[i][j];
            // Add the noise if the value is above a threshold (0.001 * TmeasMax)
            // if (originalTrace[i][j] > 0.001 * TmeasMax)
            // {
                noisyTrace[i][j] += stdDev * distribution(gen);
                noisyTrace[i][j] = std::abs(noisyTrace[i][j]);
            // }
        }
    }

    return noisyTrace;
}

std::vector<std::vector<double>> add_noise_with_snr(const std::vector<std::vector<double>> &originalTrace, int N, double desired_snr_db)
{
    std::vector<std::vector<double>> noisyTrace(N, std::vector<double>(N));

    // Seed for random number generation
    std::random_device rd;
    std::mt19937 gen(rd());

    // Calculate power of the signal
    double signal_power = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            signal_power += std::pow(originalTrace[i][j], 2);
        }
    }
    signal_power /= (N * N);

    // Calculate power of the noise needed to achieve the desired SNR
    double noise_power = signal_power / std::pow(10, desired_snr_db / 10);

    // Calculate standard deviation of the noise
    double stdDev = std::sqrt(noise_power);

    // Distribution for random noise
    std::normal_distribution<> distribution(0, stdDev);

    // Add noise to each element in the matrix
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            noisyTrace[i][j] = originalTrace[i][j] + distribution(gen);
            noisyTrace[i][j] = std::abs(noisyTrace[i][j]);
        }
    }

    return noisyTrace;
}

#endif // UTILS_INCLUDED