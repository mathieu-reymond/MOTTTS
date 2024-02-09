#ifndef AI_TOOLBOX_BANDIT_MO_EXPERIENCE_HEADER_FILE
#define AI_TOOLBOX_BANDIT_MO_EXPERIENCE_HEADER_FILE

#include <AIToolbox/Bandit/Types.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class computes averages and counts for a Bandit problem.
     *
     * This class can be used to compute the averages and counts for all
     * actions in a Bandit problem over time.
     */
    class MOExperience {
        public:
            using VisitsTable = std::vector<unsigned long>;

            /**
             * @brief Basic constructor.
             *
             * @param A The size of the action space.
             */
            MOExperience(size_t A, size_t W);

            /**
             * @brief This function updates the reward matrix and counts.
             *
             * @param a The action taken.
             * @param rew The rewards obtained.
             */
            void record(size_t a, const Vector & rew);

            /**
             * @brief This function resets the QFunction and counts to zero.
             */
            void reset();

            /**
             * @brief This function returns the number of times the record function has been called.
             *
             * @return The number of recorded timesteps.
             */
            unsigned long getTimesteps() const;

            /**
             * @brief This function returns a reference to the internal reward matrix.
             *
             * The reward matrix contains the current average rewards computed for each action.
             *
             * @return A reference to the internal reward matrix.
             */
            const Matrix2D & getRewardMatrix() const;

            /**
             * @brief This function returns a reference for the counts for the actions.
             *
             * @return A reference to the counts of the actions.
             */
            const VisitsTable & getVisitsTable() const;

            /**
             * @brief This function returns the estimated squared distance of the samples from the mean.
             *
             * The retuned values estimate sum_i (x_i - mean_x)^2 for the
             * rewards of each action. Note that these values only have
             * meaning when the respective action has at least 2 samples.
             *
             * @return A reference to the estimated square distance from the mean.
             */
            const Matrix2D & getM2Matrix() const;

            /**
             * @brief This function returns the size of the action space.
             *
             * @return The size of the action space.
             */
            size_t getA() const;

            /**
             * @brief This function returns the number of objectives.
             *
             * @return The number of objectives.
             */
            size_t getW() const;

        private:
            Matrix2D q_;
            Matrix2D M2s_;
            VisitsTable counts_;
            unsigned long timesteps_;
    };
}

#endif

