#ifndef AI_TOOLBOX_UTILS_CLUSTERING_HEADER_FILE
#define AI_TOOLBOX_UTILS_CLUSTERING_HEADER_FILE

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/digamma.hpp>

#include <Eigen/Cholesky>

#include <AIToolbox/Impl/Logging.hpp>

#include <AIToolbox/Types.hpp>
#include <Utils.hpp>
#include <iostream>

namespace AIToolbox {
    /**
     * @brief This function detect clusters in data using kMeans.
     *
     * Given a set of datapoints, this function returns the means and
     * assignments of the clusters that best describe the data.
     *
     * Note that kMeans is not necessarily guaranteed to always converge to the
     * most optimal solution; the final result depends on the initialization
     * (which here is done randomly using the provided random generator).
     *
     * @param clusters The number of clusters to use for the data.
     * @param data The datapoints to cluster (N x D)
     * @param gen A random number generator to initialize the clusters' means.
     *
     * @return The means for each cluster (K x D), and the assigned cluster id for each input point.
     */
    template <typename Gen, typename M2D>
    std::pair<std::vector<unsigned>, Matrix2D> kMeans(size_t clusters, const Eigen::MatrixBase<M2D> & data, Gen &) {
        std::pair<std::vector<unsigned>, Matrix2D> retval{
            data.rows(),
            {clusters, data.cols()}
        };
        auto & [assignments, means] = retval;
        means.setZero();

        Vector counts(clusters);
        counts.setZero();

        auto oldAssignments = assignments;

        // ### INITIALIZE CLUSTER CENTERS ###

        if (clusters != 2) throw 5;

        // Initialize from empirical means of the two sets of input data.
        means(0, 0) = data.topRows(data.rows()/2).sum() / (data.rows()/2);
        means(1, 0) = data.bottomRows(data.rows()/2).sum() / (data.rows()/2);
        std::cout << data.transpose() << '\n';
        std::cout << "Starting values = " << means.transpose() << '\n';

        AI_LOGGER(AI_SEVERITY_DEBUG, "### Running KMEANS (" << clusters << " clusters) ###");
        AI_LOGGER(AI_SEVERITY_DEBUG, "Assigned points: " << counts.transpose());
        AI_LOGGER(AI_SEVERITY_DEBUG, "Initial clusters initialized:\n" << means);

        while (true) {
            // Assign points to closest cluster.
            auto dists = makeDist(data, means);
            for (auto p = 0; p < data.rows(); ++p)
                dists.row(p).minCoeff(&assignments[p]);

            if (assignments == oldAssignments) break;
            oldAssignments = assignments;

            // Recompute cluster means
            means.setZero();
            counts.setZero();
            for (auto p = 0; p < data.rows(); ++p) {
                const auto c = assignments[p];
                means.row(c) += data.row(p);
                counts[c] += 1;
            }
            // Normalize them
            means.array().colwise() /= counts.array().cwiseMax(1.0); // cwiseMax to prevent division by zero

            AI_LOGGER(AI_SEVERITY_DEBUG, "Assigned points: " << counts.transpose());
            AI_LOGGER(AI_SEVERITY_DEBUG, "New clusters computed:\n" << means);
        }
        std::cout << "Ending values = " << means.transpose() << '\n';

        return retval;
    }

    /**
     * @brief This function detect clusters in data using Expectation-Maximization.
     *
     * This function clusters data using multinomial Normal distributions.
     *
     * An initial clustering is done using kMeans. Then we progressively
     * iterate on the mean-covariance parameters of each Normal, until
     * convergence for the log-likelihood.
     *
     * Note that the result constitutes a mixture of Gaussians distribution,
     * which can be sampled from to mimic the original dataset.
     *
     * @param clusters The number of clusters to use for the data.
     * @param data The datapoints to cluster (N x D)
     * @param gen A random number generator to initialize the clusters' means.
     *
     * @return The means for each cluster (K x D), the covariances for each cluster (K x (D x D)), and the weights of each cluster.
     */
    template <typename Gen, typename M2D>
    GMM EM(size_t clusters, const Eigen::MatrixBase<M2D> & data, Gen & gen) {
        constexpr double covarianceDiag = 1e-6;

        auto [assignments, means] = kMeans(clusters, data, gen);

        // Set initial mixing coefficients equal to fraction of assigned points to
        // each cluster.
        Matrix2D responsibilities(clusters, data.rows());
        responsibilities.setZero();

        Vector pi(clusters);
        pi.setZero();

        for (int p = 0; p < data.rows(); ++p) {
            ++pi[assignments[p]];
            // We temporarily use the responsibilities as a binary array so we can
            // more easily compute the initial covariances.
            ++responsibilities(assignments[p], p);
        }
        // Prevent division by zero
        pi = pi.cwiseMax(10.0 * std::numeric_limits<double>::min());

        // Set initial covariances equal to the covariance of each cluster of points.
        Matrix2D covHelper(data.rows(), data.cols());
        Matrix3D covs(clusters), precs(clusters);
        for (size_t c = 0; c < clusters; ++c) {
            // N x D matrix
            covHelper = data.rowwise() - means.row(c);
            // D x D matrix
            covs[c] = (covHelper.array().colwise() * responsibilities.row(c).transpose().array()).matrix().transpose() * covHelper / pi[c];
            covs[c].diagonal().array() += covarianceDiag;
        }

        // Normalize responsibilities
        pi /= data.rows();

        double logLikelihood = -std::numeric_limits<double>::infinity();

        while (true) {
            // ### E-step ###
            for (size_t c = 0; c < clusters; ++c) {
                // This is basically Sigma^-1/2
                precs[c] = covs[c].llt().matrixL().solve(Matrix2D::Identity(data.cols(), data.cols())).transpose();
            }

            // Compute pi * Normal(mean, cov) for every datapoint/cluster (numerator of 9.23)
            for (size_t c = 0; c < clusters; ++c) {
                covHelper = (data * precs[c]).rowwise() - means.row(c) * precs[c];
                responsibilities.row(c) = precs[c].diagonal().prod() * (-0.5 * covHelper.array().square().rowwise().sum().transpose()).exp();
            }
            responsibilities.array().colwise() *= pi.array();

            // Compute new log-logLikelihood (9.28)
            const double newLogLikelihood = responsibilities.colwise().sum().array().log().sum();

            // If we didn't improve by much, end EM
            if (newLogLikelihood - logLikelihood < 0.001) break;
            logLikelihood = newLogLikelihood;

            // Normalize responsibilities across clusters (denominator of 9.23)
            responsibilities.array().rowwise() /= responsibilities.colwise().sum().array();

            // ### M-step ###
            pi = responsibilities.rowwise().sum(); // N_k (9.27)
            pi = pi.cwiseMax(10.0 * std::numeric_limits<double>::min()); // Prevent division by zero

            means = responsibilities * data;
            means.array().colwise() /= pi.array(); // (9.24)

            for (size_t c = 0; c < clusters; ++c) {
                // N x D matrix
                covHelper = data.rowwise() - means.row(c);
                // D x D matrix (9.25)
                covs[c] = (covHelper.array().colwise() * responsibilities.row(c).transpose().array()).matrix().transpose() * covHelper / pi[c];
                covs[c].diagonal().array() += covarianceDiag;
            }

            pi /= pi.sum(); // (9.26)
        }

        return {pi, means, covs};
    }

    /**
     * @brief This function detect clusters in data using variational Expectation-Maximization.
     *
     * This function clusters data using multinomial Normal distributions.
     * The advantage w.r.t. the standard Expectation-Maximization algorithm is
     * that the number of clusters we set is an upper-bound, rather than an
     * exact amount.
     *
     * VEM is then able to use priors and Bayesian logic to infer the most
     * likely number of clusters needed. We have a Dirichlet prior on the
     * weights of the distributions, which makes more clusters less likely, and
     * thus they are only added to the final result if needed.
     *
     * The priors are currently baked in, using the default settings from scikit.
     *
     * An initial clustering is done using kMeans. Then we progressively
     * iterate on the mean-covariance parameters of each Normal, until
     * convergence for the log-likelihood.
     *
     * Note that the result constitutes a mixture of Gaussians distribution,
     * which can be sampled from to mimic the original dataset.
     *
     * @param clusters The number of clusters to use for the data.
     * @param data The datapoints to cluster (N x D)
     * @param gen A random number generator to initialize the clusters' means.
     *
     * \sa EM()
     *
     * @return The means for each cluster (K x D), the covariances for each
     * cluster (K x (D x D)), and the weights of each cluster. Extremely low
     * weights represent "discarded" clusters.
     */
    template <typename Gen, typename M2D>
    GMM variationalEM(size_t clusters, const Eigen::MatrixBase<M2D> & data, Gen & gen, double alpha0 = 0.0, double beta0 = 0.0) {
        Matrix2D covHelper(data.rows(), data.cols());

        // To follow the code it might be useful to grab a copy of
        //
        //     "Pattern Recognition and Machine Learning"
        //
        // by Christopher Bishop, chapter 10.
        // The variable names follow the book's equations.
        //
        // Scikit's implementation was used as reference.
        //
        // These priors are the default parameter settings in scikit's code.
        constexpr double covarianceDiag = 1e-6;
        const unsigned D = data.cols(); // Dimensions
        const double v0 = D; // Degrees of freedom
        if (alpha0 == 0.0) alpha0 = 1.0 / clusters; // Dirichlet prior
        if (beta0 == 0.0) beta0 = 1.0; // Precision prior
        const Vector means0 = data.colwise().mean(); // Mean prior (from data)
        covHelper = data.rowwise() - means0.transpose();
        // const Matrix2D cov0 = (covHelper.transpose() * covHelper) / (data.rows() - 1.0); // Covariance prior (from data)
        const Matrix2D cov0 = Matrix2D::Identity(D, D) * 0.5;

        AI_LOGGER(AI_SEVERITY_DEBUG, "### RUNNING VEM (" << clusters << ") ###");
        AI_LOGGER(AI_SEVERITY_DEBUG,
            "PRIORS:" <<
            "\nD      = " << D <<
            "\nv0     = " << v0 <<
            "\nalpha0 = " << alpha0 <<
            "\nbeta0  = " << beta0 <<
            "\nmeans0 = " << means0.transpose() <<
            "\ncov0   =\n" << cov0
        );

        // Run kmeans to get first-order approximation of the cluster's means.
        auto [assignments, means] = kMeans(clusters, data, gen);

        // Params (note that we reuse the covs/means/Nk variables,
        // respectively, to store S/xk/tmp so we allocate less memory)
        Matrix2D responsibilities(clusters, data.rows());
        Matrix3D covs(clusters), precs(clusters);
        Vector Nk, alpha, beta, v;

        // We temporarily use the responsibilities as a binary array so we can
        // more easily compute the initial covariances.
        responsibilities.setZero();
        for (int p = 0; p < data.rows(); ++p)
            ++responsibilities(assignments[p], p);

        double lowerBound = -std::numeric_limits<double>::max();
        int maxIter = 100, iter = 1;
        while (true) {
            // ### M-step ###
            //
            // These are equations 10.51, 10.52 and 10.53 in the book.

            // - Compute "naive" parameters as EM would do.
            Nk = responsibilities.rowwise().sum();
            Nk = Nk.cwiseMax(10.0 * std::numeric_limits<double>::min()); // Prevent division by zero
            means = (responsibilities * data).array().colwise() / Nk.array();

            for (size_t c = 0; c < clusters; ++c) {
                // N x D matrix
                covHelper = data.rowwise() - means.row(c);
                // D x D matrix
                covs[c] = (covHelper.array().colwise() * responsibilities.row(c).transpose().array()).matrix().transpose() * covHelper / Nk[c];
                // Regularization for llt solving later (from scikit)
                covs[c].diagonal().array() += covarianceDiag;
            }

            // Now we integrate the Bayesian prior in the parameters.
            // These are equations 10.58, 10.60, 10.61, 10.62 and 10.63.
            //
            // Note that, differently from the book, we normalize the
            // covariances. Why? Because scikit does, and so I trust them that
            // it makes sense. Additionally, we obtain a partial inverse by
            // Cholesky decomposition. These changes ultimately result in
            // some minor adjustments during the E-step.

            // - Compute dirichlet params, scaling factor and degrees of freedom.
            alpha = Nk.array() + alpha0;
            beta = Nk.array() + beta0;
            v = Nk.array() + v0;

            // - Compute covariances
            for (size_t c = 0; c < clusters; ++c) {
                covHelper.row(0) = means.row(c) - means0.transpose(); // xk - m0
                covs[c] = cov0 + Nk[c] * covs[c] + (Nk[c] * beta0) / beta[c] * (covHelper.row(0).transpose() * covHelper.row(0));
                // Normalize covariance (from scikit)
                covs[c] /= v[c];
            }

            // - Compute (~sqrt) precisions using LLT decomposition
            for (size_t c = 0; c < clusters; ++c) {
                precs[c] = covs[c].llt().matrixL().solve(Matrix2D::Identity(D, D)).transpose();
            }

            // - Compute new means
            means = ((means.array().colwise() * Nk.array()).rowwise() + beta0 * means0.array().transpose()).colwise() / beta.array();

            // ### E-step ###

            // Estimate log(q(Z)), i.e. log-responsibilities
            //
            // This is done using Equation 10.46 in Bishop's book, and the
            // expectations taken from equations 10.64, 10.65 and 10.66.
            //
            // We reshuffle the terms to do the minimum amount of work possible, so
            // we first compute the constant terms, then the per-K terms, and
            // finally the full responsibility matrix.

            // This we could sum first, but we'd have to multiply by 2 (since
            // constPart is going to be multiplied by 0.5 inside the for loop). So
            // we leave as-is and we subtract it in the loop.
            const double digammaTot = boost::math::digamma(alpha.sum());

            // CONST PART:
            //
            // - From 10.65: +D * log(2.0)
            // - From 10.46: -D * log(2 * pi)
            const double constPart = D * (std::log(2.0) - std::log(2.0 * boost::math::constants::pi<double>()));
            // K PART:
            //
            // - From scikit, due to our precision normalization: D * log(v)
            // - From 10.64: D / Bk
            Nk = constPart - D * v.array().log() - D / beta.array();
            for (size_t c = 0; c < clusters; ++c) {
                // - From E[Lambdak]: sum_d digamma(...)
                for (size_t d = 0; d < D; ++d)
                    Nk[c] += boost::math::digamma(0.5 * (v[c] - d));

                // Multiply by 1/2 all previous terms; the remaining ones go as-is.
                Nk[c] *= 0.5;

                // - From E[Lambdak]: ln(det(precision)). Note that this is already half of the true determinant of the precision.
                Nk[c] += precs[c].diagonal().array().log().sum();

                // E[ln(PI)]
                Nk[c] += boost::math::digamma(alpha[c]) - digammaTot;
            }
            for (size_t c = 0; c < clusters; ++c) {
                // From 10.64 (note that we skip vk, due to the normalization)
                // I tried also
                //
                //     covHelper = (data.rowwise() - means.row(c)) * precs[c];
                //
                // But it was slower. Not sure why...
                covHelper = (data * precs[c]).rowwise() - means.row(c) * precs[c];
                responsibilities.row(c) = -0.5 * covHelper.array().square().rowwise().sum().transpose();
            }
            responsibilities.colwise() += Nk;
            // LogSumExp per column to normalize
            auto maxTrick = responsibilities.colwise().maxCoeff();
            responsibilities.array().rowwise() -= maxTrick.array() + (responsibilities.rowwise() - maxTrick).array().exp().colwise().sum().log();

            // Finally, compute the log-likelihood lower bound. In the book,
            // this is described in equations from 10.70 to 10.77.
            //
            // At the same time, we actually use a simplified computation from
            // scikit. I have tried to do this derivation by hand but
            // ultimately I was not able to, and could not find an alternative
            // source for this. At the same time, it works, so we use it.
            //
            // These terms don't really appear like this in scikit's code; I
            // have rearranged them to (hopefully) do the sum as efficiently as
            // possible.

            double prevLowerBound = lowerBound;
            lowerBound = 0.5 * D * (std::log(2.0) * v.sum() - beta.array().log().sum());
            lowerBound -= (responsibilities.array().exp() * responsibilities.array()).sum();
            lowerBound -= std::lgamma(alpha.sum());
            for (size_t c = 0; c < clusters; ++c) {
                lowerBound += std::lgamma(alpha[c]);
                for (size_t d = 0; d < D; ++d)
                    lowerBound += std::lgamma(0.5 * (v[c] - d));
                lowerBound += v[c] * (precs[c].diagonal().array().log().sum() - 0.5 * D * std::log(v[c]));
            }

            AI_LOGGER(AI_SEVERITY_DEBUG, "Current Iteration: " << iter);
            AI_LOGGER(AI_SEVERITY_DEBUG, "Lower bound improved from " << prevLowerBound << " to " << lowerBound);

            // In theory the lower bound should only increase, but this is the
            // test that scikit does, and so we comply.
            if (std::abs(lowerBound - prevLowerBound) < 0.001) break;

            // We avoid getting stuck in the loop for too long.
            ++iter;
            if (iter > maxIter) {
                AI_LOGGER(AI_SEVERITY_WARNING, "VEM: Over max number of iterations, quitting early.");
                break;
            }

            // Now exp responsibilities for the next M-step.
            responsibilities = responsibilities.array().exp();
        }
        // Normalize the final Dirichlet weights, and return.
        alpha /= alpha.sum();
        return {alpha, means, covs};
    }

    template <typename Gen, typename M2D>
    GMM kMeansToNormals(size_t clusters, const Eigen::MatrixBase<M2D> & data, Gen & gen) {
        constexpr double covarianceDiag = 1e-6;

        auto [assignments, means] = kMeans(clusters, data, gen);

        // Set initial mixing coefficients equal to fraction of assigned points to
        // each cluster.
        Matrix2D responsibilities(clusters, data.rows());
        responsibilities.setZero();

        Vector pi(clusters);
        pi.setZero();

        for (int p = 0; p < data.rows(); ++p) {
            ++pi[assignments[p]];
            // We temporarily use the responsibilities as a binary array so we can
            // more easily compute the initial covariances.
            ++responsibilities(assignments[p], p);
        }
        pi = pi.cwiseMax(10.0 * std::numeric_limits<double>::min()); // Prevent division by zero

        // Set initial covariances equal to the covariance of each cluster of points.
        Matrix2D covHelper(data.rows(), data.cols());
        Matrix3D covs(clusters), precs(clusters);
        for (size_t c = 0; c < clusters; ++c) {
            // N x D matrix
            covHelper = data.rowwise() - means.row(c);
            if (c == 0)
                std::cout << (covHelper.array().colwise() * responsibilities.row(c).transpose().array()) <<'\n';
            // D x D matrix
            covs[c] = (covHelper.array().colwise() * responsibilities.row(c).transpose().array()).matrix().transpose() * covHelper / pi[c];
            covs[c].diagonal().array() += covarianceDiag;
        }
        pi /= data.rows();

        // std::cout << "Group 0:\n";
        // for (int p = 0; p < data.rows(); ++p) {
        //     if (assignments[p] == 0) {
        //         std::cout << data.row(p) << '\n';
        //     }
        // }
        // std::cout << "Mean:\n" << means.row(0) << '\n';
        // std::cout << "Covariance:\n" << covs[0] << '\n';
        // std::cout << "Weight:\n" << pi[0] << '\n';

        return {pi, means, covs};
    }
}

#endif
