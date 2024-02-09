#include <iostream>
#include <fstream>
#include <random>
#include <limits>
#include <iomanip>

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
// #include <Clustering.hpp>
// #include <WeightedClustering.hpp>

#include <MONormalReward.hpp>
#include <Utils.hpp>

using namespace AIToolbox;
using namespace AIToolbox::Bandit;

extern std::mt19937 globalEngine;

template <typename T>
std::ostream & operator<<(std::ostream & os, const std::vector<T> & v) {
    os << '[';
    for (size_t i = 0; i < v.size() - 1; ++i)
        os << v[i] << ", ";
    os << v.back() << ']';
    return os;
}

double entropy(const Vector & p) {
    return -p.transpose().dot(p.array().log().matrix());
}

double crowdingDistanceSort(std::vector<std::pair<double, int>> & m, const std::vector<double> & lhs, const std::vector<double> & rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        m[i].first = lhs[i];
        m[i].second = 0;
    }
    for (size_t i = 0; i < rhs.size(); ++i) {
        m[i + lhs.size()].first = rhs[i];
        m[i + lhs.size()].second = 1;
    }

    std::sort(std::begin(m), std::end(m), [](const auto & lhs, const auto & rhs){return rhs.first < lhs.first;});

    // for (const auto & p : m)
    //     std::cout << p.first << ' ' << p.second << '\n';

    double cost = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i)
        cost += (m[i].second == 0);
    for (size_t i = lhs.size(); i < m.size(); ++i)
        cost += (m[i].second == 1);

    // We don't know whether lhs is in group 0 or 1. So if we picked wrong we flip the cost around. Worst cost possible is lhs.size().
    return std::min(cost, lhs.size() * 2 - cost) / static_cast<double>(lhs.size());
}

double crowdingDistanceSortWeighted(std::vector<std::tuple<double, int, double>> & m, const std::vector<double> & lhs, const std::vector<double> & rhs, const Vector & weights) {
    // Copy input data and tag where it comes from
    for (size_t i = 0; i < lhs.size(); ++i) {
        std::get<0>(m[i]) = lhs[i];
        std::get<1>(m[i]) = 0;
        std::get<2>(m[i]) = weights[i];
    }
    for (size_t i = 0; i < rhs.size(); ++i) {
        std::get<0>(m[i + lhs.size()]) = rhs[i];
        std::get<1>(m[i + lhs.size()]) = 1;
        std::get<2>(m[i + lhs.size()]) = weights[i];
    }

    // Sort by utility value
    std::sort(std::begin(m), std::end(m), [](const auto & lhs, const auto & rhs){return std::get<0>(rhs) < std::get<0>(lhs);});

    // for (const auto & p : m)
    //     std::cout << p.first << ' ' << p.second << '\n';

    // Compute cost weighted on input weights for each particle.
    size_t i = 0;
    double cost = 0.0, weight = 0.0;
    while (weight < 1.0) {
        const auto w = std::get<2>(m[i]);
        // If we're closer on this side than on the other, just stop.
        if (1.0 - weight < (weight + w) - 1.0)
            break;
        cost += (std::get<1>(m[i]) == 0) * w;
        weight += w;
        // std::cout << weight << "(+" << w << ") ";
        ++i;
    }
    //std::cout << "\ni = " << i << "; Current w = " << weight << "; current cost = " << cost << '\n';
    while (i < m.size()) {
        const auto w = std::get<2>(m[i]);
        cost += (std::get<1>(m[i]) == 1) * w;
        weight += w;
        ++i;
    }
    // std::cout << "\nFinal w = " << weight << "; Final cost = " << cost << '\n';

    // We don't know whether lhs is in group 0 or 1. So if we picked wrong we flip the cost around. Worst cost possible is 1.0;
    return std::min(cost, 2.0 - cost);
}

Vector newtest(const Model<MONormalReward> & bandit, const UtilityFunction & uf, unsigned timesteps, unsigned queries, unsigned experiments, bool useParticles, double multiplier, unsigned particleNum) {
    Vector counters(bandit.getA());
    counters.setZero();
    Vector ufCounters(queries);
    ufCounters.setZero();

    if (!useParticles) return counters;

    const std::string methodName = "newtest";
    Matrix2D particles(particleNum, uf.getWeights().size());
    for (size_t e = 0; e < experiments; ++e) {
        writeExperiment(methodName, e, experiments, 1);

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
            ufpp.record(PARTICLEQUERY(banditp, ufpp), uf);

            --queriesLeft;
        }

        constexpr unsigned points = 500;

        std::vector<std::vector<double>> clouds1, clouds2;
        std::vector<std::vector<std::pair<double, int>>> m1;
        std::vector<std::vector<std::tuple<double, int, double>>> m2;

        clouds1.resize(banditp.getA());
        m1.resize(banditp.getA());
        for (size_t a = 0; a < banditp.getA(); ++a) {
            clouds1[a].resize(points);
            m1[a].resize(2 * points);
        }

        clouds2.resize(banditp.getA());
        m2.resize(banditp.getA());
        for (size_t a = 0; a < banditp.getA(); ++a) {
            clouds2[a].resize(ufpp.getParticles().rows());
            m2[a].resize(2 * ufpp.getParticles().rows());
        }

        unsigned queriesTaken = 0;
        while (true) {
            // if (timestep >= timesteps - 1) XXX = false;
            // std::cout << "######\n";
            // ### Compute arm crowding ###
            // - Use mean of UF, sample arms
            // - Estimate current best arm w.r.t. the mean
            const auto ufMean = ufpp.getMean();

            size_t bestArm;
            const Vector vals = (banditp.getMeans() * ufMean);
            vals.maxCoeff(&bestArm);
            // std::cout << "UF mean = " << ufMean.transpose() << " entropy = " << entropy(ufpp.getWeights()) << '\n';
            // std::cout << "Current best arm = " << bestArm << " has estimate V = " << vals[bestArm] << '\n';
            // std::cout << "SAMPLED EACH ARM: " << banditp.getCounts() << '\n';

            // Computing random sampled points for each arm.
            for (size_t a = 0; a < banditp.getA(); ++a) {
                for (size_t i = 0; i < points; ++i) {
                    clouds1[a][i] = 0.0;
                    for (size_t o = 0; o < banditp.getW(); ++o)
                        clouds1[a][i] += banditp.sampleMean(a, o, globalEngine) * ufMean[o];
                }
            }

            // ### Compute UF crowding ###
            for (size_t a = 0; a < banditp.getA(); ++a)
                for (int i = 0; i < ufpp.getParticles().rows(); ++i)
                    clouds2[a][i] = banditp.getMeans().row(a).dot(ufpp.getParticles().row(i));

            // std::cout << "BCAR - [" << clouds[bestArm][0] << ", " << clouds[bestArm].back() << "]\n";

            // Computing crowding distances.
            // double totalCAR = 0.0;
            double totalKCAR = 0.0;
            double totalKCUF = 0.0;

            #pragma omp parallel for reduction(+ : totalKCAR) num_threads(3)
            for (size_t a = 0; a < banditp.getA(); ++a) {
                if (a == bestArm) continue;

                // std::cout << "Running kmeans between arm " << bestArm << " (mean = " << banditp.getMeans().row(bestArm) << ", v = " << vals[bestArm] << ") and arm "
                //                                            << a       << " (mean = " << banditp.getMeans().row(a)       << ", v = " << vals[a]       << ")\n";
                const double c2 = crowdingDistanceSort(m1[a], clouds1[bestArm], clouds1[a]);

                // std::cout << "## - result = " << c2 << "\n";

                totalKCAR += c2 * c2;
            }

            #pragma omp parallel for reduction(+ : totalKCUF) num_threads(3)
            for (size_t a = 0; a < banditp.getA(); ++a) {
                if (a == bestArm) continue;

                // std::cout << "Running weightedsort between arm " << bestArm << " (mean = " << banditp.getMeans().row(bestArm) << ", v = " << vals[bestArm] << ") and arm "
                //                                                  << a       << " (mean = " << banditp.getMeans().row(a)       << ", v = " << vals[a]       << ")\n";
                const double c2 = crowdingDistanceSortWeighted(m2[a], clouds2[bestArm], clouds2[a], ufpp.getWeights());

                // std::cout << "## - result = " << c2 << "\n";

                totalKCUF += c2 * c2;
            }

            const double cufVsCar = totalKCUF / totalKCAR;
            const double baVsBu = static_cast<double>(timesteps-timestep) / queriesLeft;

            // std::cout << "total arm kmeans D: " << totalKCAR << "; normalized: " << totalKCAR / (banditp.getA()-1) << '\n';
            // std::cout << "total UF kmeans D: " << totalKCUF << "; normalized: " << totalKCUF / (banditp.getA()-1) << '\n';
            // std::cout << "Ratio CUF / CAR = " << cufVsCar << "; ratio of av. budget [" << (timesteps-timestep) << ", " << queriesLeft << "] = " << static_cast<double>(timesteps-timestep) / queriesLeft << '\n';

            // std::cout << "PRESS U FOR UF, A FOR ARM\n";
            char i;
            // std::cin >> i;

            if (cufVsCar > baVsBu && queriesLeft)
                i = 'u';
            else
                i = 'a';

            if (i == 'c') {
                std::cout << ufpp.getWeights() << '\n';
                continue;
            }

            // std::cout << "@ ACTION = " << i << "\n###########\n";

            // ### query UF at regular interval ###
            if (i == 'u') { // queriesLeft && currentQueryWait >= queryWait) {
                ufCounters[queriesTaken++] += timestep;
                auto q = PARTICLEQUERY(banditp, ufpp);
                const auto ans = uf.evaluateDiff(q);
                // std::cout << "@@@@@@@@@ ASKING UF QUERY = " << q.transpose() << " -> " << ans << "\n";
                ufpp.record(q, ans);

                --queriesLeft;
                continue;
            }

            if (i == 'q') break;
            if (timestep >= timesteps) break;
            ++timestep;

            // create ttts using weight estimates, not true UF
            auto mean = useParticles ? ufpp.getMean() : ufp.getMean();
            MOTopTwoThompsonSamplingPolicy ttts(exp, mean, 0.5);

            auto a = ttts.sampleAction();
            const Vector & r = bandit.sampleR(a);

            // std::cout << "@@@@@@@@@@2ASKING ARM QUERY = " << a << " -> " << r << "\n";
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
    std::cout << ufCounters.transpose() << '\n';
    return counters;
}

double crowdingDistancePoint(double v, const std::vector<double> & vec) {
    if (v < vec[0] || v > vec.back()) return 0.0;
    auto it = std::lower_bound(std::begin(vec), std::end(vec), v);
    return std::max(3.0 - (*it - *(it - 1)), 0.0) * 2/vec.size();
}

double crowdingDistanceVec(const std::vector<double> & lhs, const std::vector<double> & rhs) {
    double retval = 0.0;

    for (auto v : lhs) retval += crowdingDistancePoint(v, rhs);
    for (auto v : rhs) retval += crowdingDistancePoint(v, lhs);

    return retval / (2 * rhs.size());
}

double crowdingDistancePointWeighted(double v, const std::vector<double> & vec, const Vector & weights) {
    if (v < vec[0] || v > vec.back()) return 0.0;

    auto it = std::lower_bound(std::begin(vec), std::end(vec), v);
    auto id = std::distance(std::begin(vec), it);

    return std::max(3.0 - (*it - *(it - 1)), 0.0) * (weights[id] + weights[id-1]);
}


double crowdingDistanceVecWeighted(const std::vector<double> & lhs, const std::vector<double> & rhs, const Vector & weights) {
    double retval = 0.0;

    for (int i = 0; i < weights.size(); ++i) retval += crowdingDistancePointWeighted(lhs[i], rhs, weights) * weights[i];
    for (int i = 0; i < weights.size(); ++i) retval += crowdingDistancePointWeighted(rhs[i], lhs, weights) * weights[i];

    return retval / 2;
}

// double crowdingDistanceKmeans(Matrix2D & m, const std::vector<double> & lhs, const std::vector<double> & rhs) {
//     for (size_t i = 0; i < lhs.size(); ++i)
//         m(i, 0) = lhs[i];
//     for (size_t i = 0; i < rhs.size(); ++i)
//         m(i + lhs.size(), 0) = rhs[i];

//     const auto assignments = AIToolbox::kMeans(2, m, globalEngine).first;

//     double cost = 0.0;
//     for (size_t i = 0; i < lhs.size(); ++i)
//         cost += (assignments[i] == 0);
//     for (size_t i = lhs.size(); i < assignments.size(); ++i)
//         cost += (assignments[i] == 1);

//     // We don't know whether lhs is in group 0 or 1. So if we picked wrong we flip the cost around. Worst cost possible is lhs.size().
//     return std::min(cost, lhs.size() * 2 - cost) / static_cast<double>(lhs.size());
// }

// double crowdingDistanceKmeansWeighted(Matrix2D & m, const std::vector<double> & lhs, const std::vector<double> & rhs, const Vector & weights) {
//     for (size_t i = 0; i < lhs.size(); ++i)
//         m(i, 0) = lhs[i];
//     for (size_t i = 0; i < rhs.size(); ++i)
//         m(i + lhs.size(), 0) = rhs[i];

//     const Vector dw = weights.replicate(2, 1);
//     const auto km = AIToolbox::weightedKMeans(2, m, dw, globalEngine);
//     const auto assignments = km.first;
//     const auto means = km.second;

//     double costA = 0.0, costB = 0.0;
//     for (size_t i = 0; i < lhs.size(); ++i) {
//         costA += (assignments[i] == 0) * dw[i];
//         costB += (assignments[i] == 1) * dw[i];
//     }
//     for (size_t i = lhs.size(); i < assignments.size(); ++i) {
//         costA += (assignments[i] == 1) * dw[i];
//         costB += (assignments[i] == 0) * dw[i];
//     }
//     // std::cout << "MEANS:\n" << means << '\n';
//     // std::cout << m.topRows(lhs.size()).transpose() << '\n';
//     // std::cout << m.bottomRows(lhs.size()).transpose() << '\n';

//     // std::cout << "m = np.array([";
//     // for (int i = 0; i < m.rows() - 1; ++i)
//     //     std::cout << m(i, 0) << ", ";
//     // std::cout << m(m.rows()-1, 0) << "])\n";

//     // std::cout << "w = np.array([";
//     // for (int i = 0; i < dw.rows() - 1; ++i)
//     //     std::cout << dw(i) << ", ";
//     // std::cout << dw(dw.rows()-1) << "])\n";

//     // for (size_t i = 0; i < lhs.size(); ++i)
//     //     std::cout << assignments[i] << ' ';
//     // std::cout << '\n';
//     // for (size_t i = lhs.size(); i < assignments.size(); ++i)
//     //     std::cout << assignments[i] << ' ';
//     // std::cout << '\n';
//     // std::cout << weights.transpose() << '\n';
//     // std::cout << "COST A = " << costA << "; COST B = " << costB << '\n';

//     return std::min(costA, costB);
// }

