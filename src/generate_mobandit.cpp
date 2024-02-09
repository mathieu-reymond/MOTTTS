#include <AIToolbox/Types.hpp>
#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Prune.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <IO.hpp>
#include <string>
#include <fstream>

using namespace AIToolbox;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wstringop-truncation"

#define CONVHULL_3D_ENABLE
#include "convhull_3d.h"

#pragma GCC diagnostic pop

bool run_baselines(const std::vector<Vector> &mu, const std::vector<Vector> &sigma, const Vector weights, size_t timesteps, size_t queries, size_t experiments, bool useParticles, double multiplier, unsigned particleNum, double threshold = 0.1, double regretDiffThreshold = 0.01);

std::tuple<std::vector<Vector>, std::vector<Vector>, Vector, uint> generate_mobandit(std::string prefix, uint nArms, uint nObjectives, bool useParticles, double multiplier, unsigned particleNum) {
    // proportion of dominated arms
    double minProportion = 0.4;
    uint timesteps = 40;
    uint experiments = 10000;
    uint queries = 5;
    uint fileCounter = 0;
    // regret diff in percentage, we compute normalized regret
    // double minRegretDiff = 0.1;
    // random generator
    std::mt19937 engine(Impl::Seeder::getSeed());
    std::uniform_real_distribution<double> sampleMu(0, 1);
    std::uniform_real_distribution<double> sampleSigma(0.05, 0.15);

    std::vector<Vector> mu(nArms);
    std::vector<Vector> sigma(nArms);
    Vector weights;

    while (true) {
        for(size_t a = 0; a < nArms; ++a) {
            mu[a].resize(nObjectives);
            sigma[a].resize(nObjectives);
            for(size_t o = 0; o < nObjectives; ++o) {
                mu[a][o] = sampleMu(engine);
                sigma[a][o] = sampleSigma(engine);
            }
        }

        // count number of dominated arms
        auto it = extractDominated(std::begin(mu), std::end(mu));
        const int nDominated = std::distance(it, std::end(mu));
        Matrix2D mu2D(nArms-nDominated+1, nObjectives);
        size_t j = 0;
        for(auto i = std::begin(mu); i < it; ++i)
            mu2D.row(j++) = *i;
        mu2D.row(j).fill(-1);

        int* outFaces; int nOutFaces;
        convhull_nd_build(mu2D.data(), mu2D.rows(), nObjectives, &outFaces, nullptr, nullptr, &nOutFaces);

        double proportionConvex = (nOutFaces-1)/static_cast<double>(mu.size());
        // bandit needs to have a minimum number of non-dominated arms
        if (proportionConvex < minProportion) continue;

        // try different weights for this bandit
        for (size_t w = 0; w < 5; ++w) {
            weights = makeRandomProbability(nObjectives, engine);
            double u1 = 0, u2 = 0;
            for(const auto& m : mu) {
                double u = m.dot(weights);
                if (u > u1) {
                    u2 = u1;
                    u1 = u;
                } else if (u > u2)
                    u2 = u;
            }
            // difference in utility between top 2 arms must not be too big
            // (too big -> too easy to differentiate),
            // nor too small (too small -> impossible to differentiate)
            double uDiff = u1-u2;
            if (uDiff < 0.02 || uDiff > 0.1) continue;

            // run baselines with this bandit and weights
            bool keep = run_baselines(mu, sigma, weights, timesteps, queries, experiments, useParticles, multiplier, particleNum);
            if (keep) {
                std::ofstream file(prefix + std::to_string(fileCounter++) + ".txt");
                writeModelParameters(file, mu, sigma, weights);
            }   
        }
    }
    return {mu, sigma, weights, timesteps};
}


