#include <Utils.hpp>

#include <iomanip>

#include <AIToolbox/Utils/Probability.hpp>

using namespace AIToolbox;
extern std::mt19937 globalEngine;

void initParticles(Matrix2D & particles) {
    if (false) {
        // RANDOM
        for (auto i = 0; i < particles.rows(); ++i)
            particles.row(i) = makeRandomProbability(particles.cols(), globalEngine);
    } else {
        // UNIFORM
        const double step = 1.0 / (particles.rows() + 1); // We don't include 1/0 & 0/1
        for (auto i = 0; i < particles.rows(); ++i) {
            const double v = step * (i+1);
            particles(i, 0) = 1.0 - v;
            particles(i, 1) = v;
        }
    }
}

std::string vectorToCSV(const Vector & v) {
    std::stringstream x;
    for (int i = 0; i < v.size() - 1; ++i)
        x << std::setprecision(8) << std::setw(9) << std::left << v[i] << ", ";
    x << std::setprecision(8) << std::setw(9) << std::left << v[v.size()-1];

    return x.str();
}

double computeRegret(const std::vector<Vector> &mu, const Vector &weights, const Vector &counters) {
    // counters sum up to 1, percentages of arm recommendations
    double bestU = -std::numeric_limits<double>::infinity();
    double worstU = std::numeric_limits<double>::infinity();
    double regret = 0;
    for (size_t i = 0; i < mu.size(); ++i) {
        const double u = mu[i].dot(weights);
        if (u > bestU) bestU = u;
        if (u < worstU) worstU = u;
        regret -= u*counters[i];
    }
    return (regret + bestU)/(bestU-worstU);
    // return (regret + bestU);
}
