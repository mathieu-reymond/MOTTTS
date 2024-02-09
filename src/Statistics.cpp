#include <Statistics.hpp>

#include <cassert>
#include <random>
#include <iostream>

static std::uniform_real_distribution<Float> pDist(0.0, 1.0);

std::pair<double, double> sampleNormalPosterior(double sampleMu, double sampleSumOfSquares, double n, std::mt19937 & rnd, double priorMu, double priorN, double alpha, double beta) {
    std::gamma_distribution<double> gd(alpha + n/2.0,
                                      1.0/(
                                           beta +
                                           sampleSumOfSquares/2.0 +
                                           (n * priorN * std::pow(sampleMu-priorMu, 2.0))/(2.0 * (n+priorN))
                                        )
                                      );
    double tau = gd(rnd);
    std::normal_distribution<double> nd(n/(n+priorN)*sampleMu + priorN/(n+priorN)*priorMu,
                                       std::sqrt(1/(tau*(priorN+n))));

    double mu = nd(rnd);
    double sigma = std::sqrt(1/tau);
    return {mu, sigma};
}

double sampleMeanPosterior(double sampleMu, double sampleSumOfSquares, unsigned n, std::mt19937 & rnd) {
    static std::student_t_distribution<double> dist;

    assert(n > 1);
    if (dist.n() != n - 1)
        dist = std::student_t_distribution<double>(n - 1);

    //     mu = est_mu - t * s / sqrt(n)
    // where
    //     s^2 = 1 / (n-1) * sum_i (x_i - est_mu)^2
    // and
    //     t = student_t sample with n-1 degrees of freedom
    double v = dist(rnd);
    // std::cout << "dr[" << n << "-" << dist.n() << "](" << v << ")";
    return sampleMu + v * std::sqrt(sampleSumOfSquares / (n * (n - 1)));
}

void sampleMultivariateNormalInline(const Vector & mu, const Eigen::LLT<Matrix2D> & llt, Vector & out, std::mt19937 & rnd) {
    static std::normal_distribution<double> dist;
    auto randN = [&rnd](){ return dist(rnd); };

    out = mu + llt.matrixL() * Vector::NullaryExpr(mu.size(), randN);
}

void sampleMultivariateNormalInline(const Vector & mu, const Matrix2D & cov, Vector & out, std::mt19937 & rnd) {
    // LLT is basically to square root the variance (since we need the std).
    sampleMultivariateNormalInline(mu, cov.llt(), out, rnd);
}

void sampleNormalizedMultivariateNormalInline(const Vector & mu, const Eigen::LLT<Matrix2D> & llt, Vector & out, std::mt19937 & rnd) {
    sampleMultivariateNormalInline(mu, llt, out, rnd);

    // Normalize (note the abs!)
    out.array() /= out.array().abs().sum();
}

void sampleNormalizedMultivariateNormalInline(const Vector & mu, const Matrix2D & cov, Vector & out, std::mt19937 & rnd) {
    sampleNormalizedMultivariateNormalInline(mu, cov.llt(), out, rnd);
}

size_t sampleProbability(const Vector& in, std::mt19937& generator) {
    const size_t D = in.size();
    Float p = pDist(generator);

    for ( size_t i = 0; i < D; ++i ) {
        if ( in[i] > p ) return i;
        p -= in[i];
    }
    return D-1;
}

Vector makeRandomProbability(const size_t S, std::mt19937 & generator) {
    Vector b(S);
    Float * bData = b.data();
    // The way this works is that we're going to generate S-1 numbers in
    // [0,1], and sort them with together with an implied 0.0 and 1.0, for
    // a total of S+1 numbers.
    //
    // The output will be represented by the differences between each pair
    // of numbers, after sorting the original vector.
    //
    // The idea is basically to take a unit vector and cut it up into
    // random parts. The size of each part is the value of an entry of the
    // output.

    // We must set the first element to zero even if we're later
    // overwriting it. This is to avoid bugs in case the input S is one -
    // in which case we should return a vector with a single element
    // containing 1.0.
    bData[0] = 0.0;
    for ( size_t s = 0; s < S-1; ++s )
        bData[s] = pDist(generator);

    // Sort all but the implied last 1.0 which we'll add later.
    std::sort(bData, bData + S - 1);

    // For each number, keep track of what was in the vector there, and
    // transform it into the difference with its predecessor.
    Float helper1 = bData[0], helper2;
    for ( size_t s = 1; s < S - 1; ++s ) {
        helper2 = bData[s];
        bData[s] -= helper1;
        helper1 = helper2;
    }
    // The final one is computed with respect to the overall sum of 1.0.
    bData[S-1] = 1.0 - helper1;

    return b;
}
