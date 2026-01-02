/**
 * @file testPulseClass.cpp
 * @author Víctor Loras Herrero
 * @brief Test database creation
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "utils.hpp"
#include "fourier.hpp"
#include "pulse.hpp"
#include "cnpy.h"
#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <string>
#include <fstream>
#include <random>
#include <iomanip>

int main()
{
    

    double t0 = 0;
    int    N = 64;
    double signalDuration = 1.368942071984176e-12;  // seconds (T_train)
    double deltaT = signalDuration / N;            

    FourierTransform ft(N, deltaT, t0);

    Pulse generatedPulse(ft);
    std::vector<std::vector<double>> Tmn;
    std::vector<std::complex<double>> field;

    // DB parameters
    int numberOfPulses = 10000;         
    double initialTBP  = 0.5;
    double finalTBP    = 0.76;

    // random TBP on a 0.01 grid 
    int tbp_min = static_cast<int>(std::round(initialTBP * 100.0));
    int tbp_max = static_cast<int>(std::round(finalTBP   * 100.0));

    std::mt19937 rng(static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    ));
    std::uniform_int_distribution<int> pickTBP(tbp_min, tbp_max);

    //  pre-allocate output arrays for .npy saving
    std::vector<double> traces(static_cast<size_t>(numberOfPulses) * N * N);   // X: [P, N, N]
    std::vector<double> profiles(static_cast<size_t>(numberOfPulses) * 2 * N); // Y: [P, 2N]


    std::cout << "Generating " << numberOfPulses << " random pulses with TBP in [" << initialTBP << ", " << finalTBP << "] " << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < numberOfPulses; ++n)
    {
        
        double currentTBP = pickTBP(rng) / 100.0;

        generatedPulse.randomPulse(currentTBP);
        field = generatedPulse.getField();

        double maxAmp = 0.0;
        for (const auto& z : field) {
            double a = std::abs(z);      
            if (a > maxAmp) maxAmp = a;
        }
        if (maxAmp == 0.0) maxAmp = 1.0;  // guard against a zero field

   
    size_t baseY = static_cast<size_t>(n) * static_cast<size_t>(2 * N);
    for (int k = 0; k < N; ++k) {
        profiles[baseY + k]     = std::real(field[k]) / maxAmp; // Real part
        profiles[baseY + N + k] = std::imag(field[k]) / maxAmp; // Imaginary part
    }

 
    Tmn = trace(field, ft.t, deltaT);
    double Tmax = 0.0;
    for (const auto& row : Tmn)
        for (double v : row)
            if (v > Tmax) Tmax = v;

    const double invT = (Tmax > 0.0) ? (1.0 / Tmax) : 1.0;

    size_t baseX = static_cast<size_t>(n) * static_cast<size_t>(N) * static_cast<size_t>(N);
    for (int k = 0; k < N; ++k) {
        for (int l = 0; l < N; ++l) {
            traces[baseX + static_cast<size_t>(k) * N + l] = Tmn[k][l] * invT; 
        }
    }

    }

    //save the dataset to .npy files 
    std::vector<size_t> Xshape = { static_cast<size_t>(numberOfPulses), static_cast<size_t>(N), static_cast<size_t>(N) };
    std::vector<size_t> Yshape = { static_cast<size_t>(numberOfPulses), static_cast<size_t>(2 * N) };

    std::string xfile = "FROG_T_N" + std::to_string(N) + "_P" + std::to_string(numberOfPulses) + ".npy";
    std::string yfile = "E_profile_N" + std::to_string(N) + "_P" + std::to_string(numberOfPulses) + ".npy";

    cnpy::npy_save(xfile, traces.data(), Xshape, "w");   // X: shape [P, N, N]
    cnpy::npy_save(yfile, profiles.data(), Yshape, "w"); // Y: shape [P, 2N]



    // End the timer
    auto endTime = std::chrono::high_resolution_clock::now();

   
    // Compute the elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Print the elapsed time
    std::cout << "Elapsed time generating database: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}