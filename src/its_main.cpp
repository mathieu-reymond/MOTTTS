#include <iostream>
#include <fstream>
#include <random>
#include <limits>
#include <iomanip>
#include <filesystem>
#include <stdexcept>

#include <CommandLineParsing.hpp>
#include <MOTTTS.hpp>
#include <AIToolbox/Bandit/Model.hpp>
#include <AIToolbox/Tools/Statistics.hpp>
#include <MOThompsonSamplingPolicy.hpp>
#include <Statistics.hpp>
#include <UtilityFunction.hpp>
#include <UtilityFunctionPosterior.hpp>
#include <UtilityFunctionParticlePosterior.hpp>
#include <MOBanditNormalPosterior.hpp>
#include <IO.hpp>
#include <MONormalReward.hpp>
#include <Utils.hpp>

using namespace AIToolbox;
using namespace Bandit;

std::mt19937 globalEngine;

std::tuple<Matrix2D, Matrix2D> its(const Model<MONormalReward> & bandit, const UtilityFunction & uf, unsigned timesteps, unsigned experiments, bool useParticles, double multiplier, unsigned particleNum) {
    Matrix2D pulls(experiments, timesteps);
    Matrix2D queries(experiments, timesteps);
    pulls.setZero();
    queries.setZero();

    static Vector sampledW0, sampledW1, meanR0, meanR1;
    static std::mt19937 rnd(AIToolbox::Impl::Seeder::getSeed());

    sampledW0.resize(uf.getWeights().size());
    sampledW1.resize(uf.getWeights().size());
    meanR0.resize(uf.getWeights().size());
    meanR1.resize(uf.getWeights().size());

    const std::string methodName = "its";
    Matrix2D particles(particleNum, uf.getWeights().size());
    for (size_t e = 0; e < experiments; ++e) {
        writeExperiment(methodName, e, experiments);

        MOExperience exp(bandit.getA(), 2);

        initParticles(particles);

        UtilityFunctionParticlePosterior ufpp(particles, 0.05);
        UtilityFunctionPosterior ufp(uf.getWeights().size(), multiplier);
        MOBanditNormalPosterior banditp(bandit.getA(), uf.getWeights().size());

        // manually pull each arm twice, so we can ask queries afterwards
        unsigned timestep = 0;
        for (size_t a = 0; a < bandit.getA(); ++a) {
            for (size_t n = 0; n < 2; ++n) {
                const Vector & r = bandit.sampleR(a);

                exp.record(a, r);
                banditp.record(a, r);

                pulls.row(e)[timestep] = a;

                ++timestep;
            }
        }

        // 2 initial queries
        for (size_t q = 0; q < 2; ++q) {
            if (useParticles) {
                ufpp.record(PARTICLEQUERY(banditp, ufpp), uf);
            } else {
                ufp.record(suggestQueryThompson(banditp, ufp), uf);
                ufp.optimize();
            }
            // asked 2 queries at the same timestep
            queries.row(e)[timestep-1] = 2;
        }

        // don't start from zero, already used part of the budget
        for (; timestep < timesteps; ++timestep) {
            // sample 2 different weights from UF posterior
            if (useParticles) {
                size_t pId = sampleProbability(ufpp.getWeights(), rnd);
                sampledW0 = particles.row(pId);
                pId = sampleProbability(ufpp.getWeights(), rnd);
                sampledW1 = particles.row(pId);
            } else {
                sampleNormalizedMultivariateNormalInline(ufp.getMean(), ufp.getCovLLT(), sampledW0, rnd);
                sampleNormalizedMultivariateNormalInline(ufp.getMean(), ufp.getCovLLT(), sampledW1, rnd);
            }
            // use TS to sample action, for each of the 2 sampled weights
            auto a0 = MOThompsonSamplingPolicy(exp, sampledW0).sampleAction();
            auto a1 = MOThompsonSamplingPolicy(exp, sampledW1).sampleAction();

            // MOThompsonSamplingPolicy ts(exp, mean);
            // auto a = ts.sampleAction();
            
            // pull arm using first TS sample
            const Vector & r = bandit.sampleR(a0);
            exp.record(a0, r);

            // if both sampled arms are different, ask a query
            if (a0 != a1) {
                // query is based on mean of pulled arm-rewards
                meanR0 = exp.getRewardMatrix().row(a0);
                meanR1 = exp.getRewardMatrix().row(a1);
                if (useParticles) {
                    ufpp.record(meanR0-meanR1, uf);
                } else {
                    ufp.record(meanR0-meanR1, uf);
                    ufp.optimize();
                }
                queries.row(e)[timestep] = 1;
            }
            pulls.row(e)[timestep] = a0;
        }
    }
    return {pulls, queries};
}

void run_its(const std::vector<Vector> &mu,
             const std::vector<Vector> &sigma,
             const Vector weights,
             size_t timesteps,
             size_t experiments,
             [[maybe_unused]] bool useParticles,
             [[maybe_unused]] double multiplier,
             [[maybe_unused]] unsigned particleNum) {
    // Construct Model
    std::vector<std::tuple<Vector, Vector>> armParams;
    for (size_t i = 0; i < mu.size(); ++i)
        armParams.emplace_back(mu[i], sigma[i]);
    Model<MONormalReward> bandit(std::move(armParams));

    double bestU = -std::numeric_limits<double>::infinity();
    {
        std::cout << "Arm utilities: ";
        for (const auto & m : mu) {
            const double u = m.dot(weights);
            std::cout << std::setw(9) << std::left << u << ' ';

            bestU = std::max(u, bestU);
        }
        std::cout << "\nArm regrets:   ";
        for (const auto & m : mu) {
            const double u = m.dot(weights);
            std::cout << std::setw(9) << std::left << (bestU - u) << ' ';
        }
        std::cout << '\n';
    }
    // create utility function
    UtilityFunction uf(weights, 0);

    // ITS on the bandit
    auto [pulls, queries] = its(bandit, uf, timesteps, experiments, useParticles, multiplier, particleNum);

    // compute regret
    Matrix2D regret(experiments, timesteps);
    Matrix2D cumulativeRegret(experiments, timesteps);
    Matrix2D cumulativeQueries(experiments, timesteps);
    regret.setZero();
    cumulativeRegret.setZero();
    cumulativeQueries.setZero();
    size_t arm;
    float currentRegret = 0.;
    size_t currentQueries = 0;
    for (size_t e = 0; e < experiments; ++e) {
        currentRegret = 0;
        currentQueries = 0;
        for (size_t t = 0; t < timesteps; ++t) {
            arm = pulls.row(e)[t];
            const auto & m = mu[arm];
            regret.row(e)[t] = bestU - m.dot(weights);
            currentRegret += regret.row(e)[t];
            cumulativeRegret.row(e)[t] = currentRegret;
            currentQueries += queries.row(e)[t];
            cumulativeQueries.row(e)[t] = currentQueries;
        }
    }
    std::cout << std::endl;
    auto meanCumulativeRegret = cumulativeRegret.colwise().mean();
    for (const auto& r : meanCumulativeRegret)
        std::cout << r << ",";
    std::cout << std::endl;
    auto meanCumulativeQueries = cumulativeQueries.colwise().mean();
    for (const auto& q : meanCumulativeQueries)
        std::cout << q << ",";
    std::cout << std::endl;
}


std::tuple<std::vector<Vector>, std::vector<Vector>, Vector, uint> generate_mobandit(std::string prefix, uint nArms, uint nObjectives, bool useParticles, double multiplier, unsigned particleNum);


int main(int argc, char** argv) {
    // Parameter variables.
    int seed;
    unsigned nArms;
    unsigned nObjectives;
    std::string banditfile;
    unsigned timesteps;
    unsigned experiments;
    bool useParticles;
    unsigned particleNum;

    Options options;

    // Adding options to parse.
    options.push_back(makeRequiredOption ("seed,s",       &seed,        "set the experiment's seed"));
    options.push_back(makeDefaultedOption("arms,a",       &nArms,       "number of bandit arms",   10u));
    options.push_back(makeDefaultedOption("objectives,n", &nObjectives, "objectives per arm",      2u));
    options.push_back(makeRequiredOption("file,f",       &banditfile,  "file with bandit"));
    options.push_back(makeDefaultedOption("timesteps,t",  &timesteps,   "set the num of timesteps", 40u));
    options.push_back(makeDefaultedOption("experiments,e",&experiments, "set the num of experiments", 100000u));
    options.push_back(makeDefaultedOption("useparticles,u",&useParticles, "whether to use particles", true));
    options.push_back(makeDefaultedOption("particlenum,p",&particleNum, "the number of particles to use", 1000u));

    // Parse command line options
    if (!parseCommandLine(argc, argv, options))
        return 1;

    AIToolbox::Impl::Seeder::setRootSeed(seed);
    globalEngine.seed(Impl::Seeder::getSeed());

    constexpr double multiplier = 1;

    std::cout << (useParticles ? "U" : "NOT u") << "sing particle beliefs\n";

    if(!std::filesystem::exists(banditfile)) {
        throw std::invalid_argument("invalid banditfile");
    } else {
        Vector weights;
        std::vector<Vector> mu, sigma;
        std::ifstream file(banditfile);
        std::tie(weights, mu, sigma) = parseModelParameters(file);
        run_its(mu, sigma, weights, timesteps, experiments, useParticles, multiplier, particleNum);
    }

    return 0;
}