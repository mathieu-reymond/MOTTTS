#include <iostream>
#include <fstream>
#include <random>
#include <limits>
#include <iomanip>
#include <filesystem>

#include <CommandLineParsing.hpp>
#include <MOTTTS.hpp>
#include <AIToolbox/Bandit/Policies/TopTwoThompsonSamplingPolicy.hpp>
#include <AIToolbox/Bandit/Model.hpp>
#include <AIToolbox/Tools/Statistics.hpp>
#include <Statistics.hpp>
#include <UtilityFunction.hpp>
#include <UtilityFunctionPosterior.hpp>
#include <MOBanditNormalPosterior.hpp>
#include <UtilityFunction.hpp>
#include <UtilityFunctionParticlePosterior.hpp>
#include <MOBanditNormalPosterior.hpp>
#include <IO.hpp>
#include <MONormalReward.hpp>
#include <Utils.hpp>

using namespace AIToolbox;
using namespace Bandit;

std::mt19937 globalEngine;

Vector morobin(const Model<MONormalReward> & bandit, const Vector &weights, unsigned timesteps, unsigned experiments) {
    Vector counters(bandit.getA());
    counters.setZero();
    const std::string methodName = "morobin";

    for (size_t e = 0; e < experiments; ++e) {
        writeExperiment(methodName, e, experiments);

        // Start from random action
        size_t a = std::uniform_int_distribution<size_t>(0, bandit.getA() -1)(globalEngine);
        MOExperience exp(bandit.getA(), 2);

        for (size_t i = 0; i < timesteps; ++i) {
            const Vector & r = bandit.sampleR(a);
            exp.record(a, r);

            ++a;
            if (a == bandit.getA()) a = 0;
        }

        size_t finalBest;
        (exp.getRewardMatrix() * weights).maxCoeff(&finalBest);

        ++counters[finalBest];
    }
    std::cout << '\n';

    counters /= experiments;
    return counters;
}


Vector mottts(const Model<MONormalReward> & bandit, const Vector &weights, unsigned timesteps, unsigned experiments) {
    Vector counters(bandit.getA());
    counters.setZero();

    const std::string methodName = "morobin";
    for (size_t e = 0; e < experiments; ++e) {
        writeExperiment(methodName, e, experiments);

        MOExperience exp(bandit.getA(), 2);
        MOTopTwoThompsonSamplingPolicy ttts(exp, weights, 0.5);
        // std::vector<double> armPulls(timesteps);
        // std::vector<double> armBudgets(timesteps);

        for (size_t i = 0; i < timesteps; ++i) {
            auto a = ttts.sampleAction();

            const Vector & r = bandit.sampleR(a);
            exp.record(a, r);
            // armPulls[i] = a+1;
            // armBudgets[i] = timesteps-i;
        }

        size_t finalBest;
        (exp.getRewardMatrix() * weights).maxCoeff(&finalBest);

        ++counters[finalBest];
    }
    std::cout << '\n';
    counters /= experiments;
    return counters;
}

Vector querymorobin(const Model<MONormalReward> & bandit, const UtilityFunction & uf, unsigned timesteps, unsigned queries, unsigned experiments, bool useParticles, double multiplier, unsigned particleNum) {
    Vector counters(bandit.getA());
    counters.setZero();

    Matrix2D particles(particleNum, uf.getWeights().size());

    const std::string methodName = "querymorobin";

    for (size_t e = 0; e < experiments; ++e) {
        writeExperiment(methodName, e, experiments);

        initParticles(particles);

        UtilityFunctionParticlePosterior ufpp(particles, 0.05);
        UtilityFunctionPosterior ufp(uf.getWeights().size(), multiplier);
        MOBanditNormalPosterior banditp(bandit.getA(), uf.getWeights().size());

        MOExperience exp(bandit.getA(), 2);

        // Start from random action
        size_t a = std::uniform_int_distribution<size_t>(0, bandit.getA() -1)(globalEngine);

        for (size_t i = 0; i < timesteps; ++i) {
            const Vector & r = bandit.sampleR(a);
            exp.record(a, r);
            banditp.record(a, r);

            ++a;
            if (a == bandit.getA()) a = 0;
        }

        for (size_t q = 0; q < queries; ++q) {
            if (useParticles) {
                ufpp.record(PARTICLEQUERY(banditp, ufpp), uf);
            } else {
                ufp.record(suggestQueryThompson(banditp, ufp), uf);
                ufp.optimize();
            }
        }

        size_t finalBest;
        if (useParticles)
            (exp.getRewardMatrix() * ufpp.getMean()).maxCoeff(&finalBest);
        else
            (exp.getRewardMatrix() * ufp.getMean()).maxCoeff(&finalBest);

        ++counters[finalBest];
    }
    std::cout << '\n';

    counters /= experiments;
    return counters;
}

Vector querymottts(const Model<MONormalReward> & bandit, const UtilityFunction & uf, unsigned timesteps, unsigned queries, unsigned experiments, bool useParticles, double multiplier, unsigned particleNum) {
    Vector counters(bandit.getA());
    counters.setZero();

    const std::string methodName = "querymottts";
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

                ++timestep;
            }
        }

        // std::cout << "pulled each arm twice, asking queries\n";
        for (size_t q = 0; q < queries; ++q) {
            if (useParticles) {
                ufpp.record(PARTICLEQUERY(banditp, ufpp), uf);
            } else {
                ufp.record(suggestQueryThompson(banditp, ufp), uf);
                ufp.optimize();
            }
        }

        // create ttts using weight estimates, not true UF
        auto mean = useParticles ? ufpp.getMean() : ufp.getMean();
        MOTopTwoThompsonSamplingPolicy ttts(exp, mean, 0.5);

        // don't start from zero, already used part of the budget
        for (; timestep < timesteps; ++timestep) {
            auto a = ttts.sampleAction();

            const Vector & r = bandit.sampleR(a);
            exp.record(a, r);
        }

        size_t finalBest;
        if (useParticles)
            (exp.getRewardMatrix() * ufpp.getMean()).maxCoeff(&finalBest);
        else
            (exp.getRewardMatrix() * ufp.getMean()).maxCoeff(&finalBest);

        ++counters[finalBest];
    }
    std::cout << std::endl;
    counters /= experiments;
    return counters;
}


Vector queryinterleavedmottts(const Model<MONormalReward> & bandit, const UtilityFunction & uf, unsigned timesteps, unsigned queries, unsigned experiments, bool useParticles, double multiplier, unsigned particleNum) {
    Vector counters(bandit.getA());
    counters.setZero();
    Vector ufCounters(queries);
    ufCounters.setZero();

    const std::string methodName = "queryinterleavedmottts";
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

                ++timestep;
            }
        }
        unsigned queriesLeft = queries;
        // 2 initial queries
        for (size_t q = 0; q < 2; ++q) {
            if (useParticles) {
                ufpp.record(PARTICLEQUERY(banditp, ufpp), uf);
            } else {
                ufp.record(suggestQueryThompson(banditp, ufp), uf);
                ufp.optimize();
            }

            --queriesLeft;
        }

        double queryWait = static_cast<double>(timesteps - timestep) / queriesLeft;
        double currentQueryWait = 0.0;

        unsigned queriesTaken = 0;
        while (true) {
            // query UF at regular interval
            currentQueryWait += 1.0;
            if (queriesLeft && currentQueryWait >= queryWait) {
                ufCounters[queriesTaken++] += timestep - 20;
                if (useParticles) {
                    ufpp.record(PARTICLEQUERY(banditp, ufpp), uf);
                } else {
                    ufp.record(suggestQueryThompson(banditp, ufp), uf);
                    ufp.optimize();
                }

                currentQueryWait -= queryWait;
                --queriesLeft;
            }

            if (timestep >= timesteps) break;
            ++timestep;

            // create ttts using weight estimates, not true UF
            auto mean = useParticles ? ufpp.getMean() : ufp.getMean();
            MOTopTwoThompsonSamplingPolicy ttts(exp, mean, 0.5);

            auto a = ttts.sampleAction();
            const Vector & r = bandit.sampleR(a);

            exp.record(a, r);
            banditp.record(a, r);
        }

        size_t finalBest;
        if (useParticles)
            (exp.getRewardMatrix() * ufpp.getMean()).maxCoeff(&finalBest);
        else
            (exp.getRewardMatrix() * ufp.getMean()).maxCoeff(&finalBest);

        ++counters[finalBest];
    }
    std::cout << std::endl;

    counters /= experiments;
    ufCounters /= experiments;
    return counters;
}


Vector queryendmottts(const Model<MONormalReward> & bandit, const UtilityFunction & uf, unsigned timesteps, unsigned queries, unsigned experiments, bool useParticles, double multiplier, unsigned particleNum) {
    Vector counters(bandit.getA());
    counters.setZero();

    const std::string methodName = "queryendmottts";
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

                ++timestep;
            }
        }

        // ask 2 initial queries
        // 2 initial queries
        for (size_t q = 0; q < 2; ++q) {
            if (useParticles) {
                ufpp.record(PARTICLEQUERY(banditp, ufpp), uf);
            } else {
                ufp.record(suggestQueryThompson(banditp, ufp), uf);
                ufp.optimize();
            }
        }

        // create ttts using weight estimates, not true UF
        auto mean = ufp.getMean();
        MOTopTwoThompsonSamplingPolicy ttts(exp, mean, 0.5);

        // don't start from zero, already used part of the budget
        for (; timestep < timesteps; ++timestep) {
            auto a = ttts.sampleAction();

            const Vector & r = bandit.sampleR(a);
            exp.record(a, r);
        }

        // ask all leftover queries after using pulling budget
        for (size_t q = 2; q < queries; ++q) {
            if (useParticles) {
                ufpp.record(PARTICLEQUERY(banditp, ufpp), uf);
            } else {
                ufp.record(suggestQueryThompson(banditp, ufp), uf);
                ufp.optimize();
            }
        }

        size_t finalBest;
        if (useParticles)
            (exp.getRewardMatrix() * ufpp.getMean()).maxCoeff(&finalBest);
        else
            (exp.getRewardMatrix() * ufp.getMean()).maxCoeff(&finalBest);

        ++counters[finalBest];
    }
    std::cout << std::endl;
    counters /= experiments;
    return counters;
}

bool run_baselines(const std::vector<Vector> &mu,
                   const std::vector<Vector> &sigma,
                   const Vector weights,
                   size_t timesteps,
                   [[maybe_unused]] size_t queries,
                   size_t experiments,
                   [[maybe_unused]] bool useParticles,
                   [[maybe_unused]] double multiplier,
                   [[maybe_unused]] unsigned particleNum,
                   double threshold = 0.1,
                   double regretDiffThreshold = 0.01) {
    // Construct Model
    std::vector<std::tuple<Vector, Vector>> armParams;
    for (size_t i = 0; i < mu.size(); ++i)
        armParams.emplace_back(mu[i], sigma[i]);
    Model<MONormalReward> bandit(std::move(armParams));

    {
        double bestU = -std::numeric_limits<double>::infinity();
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

    // ### METHODS THAT HAVE THE TRUE UF ###
    // MOTTTS
    auto motttsCounters = mottts(bandit, weights, timesteps, experiments);
    auto motttsRegret = computeRegret(mu, weights, motttsCounters);
    // MOROBIN
    auto morobinCounters = morobin(bandit, weights, timesteps, experiments);
    auto morobinRegret = computeRegret(mu, weights, morobinCounters);
    double regretDiff = std::abs(morobinRegret-motttsRegret);

    // Find diff in BAI between the 2 baselines
    double maxMOTTTS = 0;
    double maxMOROBIN = 0;
    for(int i = 0; i < motttsCounters.size(); ++i) {
        if (motttsCounters[i] > maxMOTTTS) maxMOTTTS = motttsCounters[i];
        if (morobinCounters[i] > maxMOROBIN) maxMOROBIN = morobinCounters[i];
    }
    double BAIDiff = std::abs(maxMOROBIN-maxMOTTTS);

    // Only run the more expensive baselines (with queries)
    // if the diff in BAI and regret is high enough
    std::cout << "BAIDiff: " << BAIDiff << ", regretDiff: " << regretDiff << "\n";

    return (BAIDiff >= threshold && regretDiff >= regretDiffThreshold);
}

void run_baselines2(const std::vector<Vector> &mu,
                   const std::vector<Vector> &sigma,
                   const Vector weights,
                   size_t timesteps,
                   size_t queries,
                   size_t experiments,
                   bool useParticles,
                   double multiplier,
                   unsigned particleNum,
                   [[maybe_unused]] double threshold = 0.1,
                   [[maybe_unused]] double regretDiffThreshold = 0.01) {
        // Construct Model
    std::vector<std::tuple<Vector, Vector>> armParams;
    for (size_t i = 0; i < mu.size(); ++i)
        armParams.emplace_back(mu[i], sigma[i]);
    Model<MONormalReward> bandit(std::move(armParams));

    {
        double bestU = -std::numeric_limits<double>::infinity();
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

    // ### METHODS THAT HAVE THE TRUE UF ###
    // MOTTTS
    auto motttsCounters = mottts(bandit, weights, timesteps, experiments);
    auto motttsRegret = computeRegret(mu, weights, motttsCounters);
    // MOROBIN
    auto morobinCounters = morobin(bandit, weights, timesteps, experiments);
    auto morobinRegret = computeRegret(mu, weights, morobinCounters);
    double regretDiff = std::abs(morobinRegret-motttsRegret);

    auto morobinqueryCounters           = querymorobin          (bandit, uf, timesteps, queries, experiments, useParticles, multiplier, particleNum);
    auto motttsqueryCounters            = querymottts           (bandit, uf, timesteps, queries, experiments, useParticles, multiplier, particleNum);
    auto motttsqueryinterleavedCounters = queryinterleavedmottts(bandit, uf, timesteps, queries, experiments, useParticles, multiplier, particleNum);
    auto motttsqueryendCounters         = queryendmottts        (bandit, uf, timesteps, queries, experiments, useParticles, multiplier, particleNum);

    auto morobinqueryregret           = computeRegret(mu, weights, morobinqueryCounters);
    double minRegret = morobinqueryregret;
    std::string method = "morobinquery";
    auto motttsqueryregret            = computeRegret(mu, weights, motttsqueryCounters);
    if (minRegret > motttsqueryregret) {
        minRegret = motttsqueryregret;
        method = "motttsquery";
    }
    auto motttsqueryendregret         = computeRegret(mu, weights, motttsqueryendCounters);
    if (minRegret > motttsqueryregret) {
        minRegret = motttsqueryendregret;
        method = "motttsqueryend";
    }
    auto motttsqueryinterleavedregret = computeRegret(mu, weights, motttsqueryinterleavedCounters);
    if (minRegret > motttsqueryinterleavedregret) {
        minRegret = motttsqueryinterleavedregret;
        method = "motttsqueryinterleaved";
    }

    std::cout << "DIFF regret: " << regretDiff << "\n";
    std::cout << "morobin,             "  << vectorToCSV(morobinCounters)                << ", " << computeRegret(mu, weights, morobinCounters)                << "\n";
    std::cout << "mottts,              "  << vectorToCSV(motttsCounters)                 << ", " << computeRegret(mu, weights, motttsCounters)                 << "\n";
    std::cout << "morobinquery,        "  << vectorToCSV(morobinqueryCounters)           << ", " << morobinqueryregret                                         << "\n";
    std::cout << "mottts-qstart,       "  << vectorToCSV(motttsqueryCounters)            << ", " << motttsqueryregret                                          << "\n";
    std::cout << "mottts-qend,         "  << vectorToCSV(motttsqueryendCounters)         << ", " << motttsqueryendregret                                       << "\n";
    std::cout << "mottts-qinterleaved, "  << vectorToCSV(motttsqueryinterleavedCounters) << ", " << motttsqueryinterleavedregret                               << "\n";

    std::cout << "BEST METHOD = " << method << " with regret " << minRegret << '\n';;
}

std::tuple<std::vector<Vector>, std::vector<Vector>, Vector, uint> generate_mobandit(std::string prefix, uint nArms, uint nObjectives, bool useParticles, double multiplier, unsigned particleNum);

int main(int argc, char** argv) {
    // Parameter variables.
    int seed;
    unsigned nArms;
    unsigned nObjectives;
    std::string banditfile;
    unsigned timesteps;
    unsigned queries;
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
    options.push_back(makeDefaultedOption("queries,q",    &queries,     "set the num of queries", 5u));
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
        for (size_t i = 0; i < 10; ++i) {
            auto [mu, sigma, weights, timesteps] = generate_mobandit(banditfile, nArms, nObjectives, useParticles, multiplier, particleNum);
            std::cout << "mu: " << mu << "\n";
            std::cout << "sigma: " << sigma << "\n";
            std::cout << "w: " << weights << "\n";
        }
    } else {
        Vector weights;
        std::vector<Vector> mu, sigma;
        std::ifstream file(banditfile);
        std::tie(weights, mu, sigma) = parseModelParameters(file);
        run_baselines2(mu, sigma, weights, timesteps, queries, experiments, useParticles, multiplier, particleNum, 0, 0);
    }

    return 0;
}
