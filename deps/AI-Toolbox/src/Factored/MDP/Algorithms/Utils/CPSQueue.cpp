#include <AIToolbox/Factored/MDP/Algorithms/Utils/CPSQueue.hpp>

#include <algorithm>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox::Factored {
    CPSQueue::CPSQueue(const DDNGraph & graph) :
            graph_(graph),
            nonZeroPriorities_(0),
            order_(graph_.getS().size()+1), nodes_(graph_.getS().size()),
            rand_(Impl::Seeder::getSeed())
    {
        const auto & S = graph_.getS();

        order_[0] = S.size();
        std::iota(std::begin(order_)+1, std::end(order_), 0);

        for (size_t i = 0; i < S.size(); ++i) {
            const auto & gn = graph_.getParentSets()[i];
            auto & n = nodes_[i];

            n.maxV = -1.0;
            n.maxA = 0;

            n.nodes.resize(gn.features.size());
            n.order.resize(gn.features.size());
            std::iota(std::begin(n.order), std::end(n.order), 0);

            for (size_t a = 0; a < gn.features.size(); ++a) {
                auto & nn = n.nodes[a];
                nn.maxV = -1.0;
                nn.maxS = 0;
                nn.priorities.resize(graph_.getPartialSize(i, a));
                nn.priorities.fill(-1.0);
            }
        }
    }

    void CPSQueue::update(size_t i, size_t a, size_t s, double p) {
        assert(p > 0.0);
        // If the priority was not assigned before, we increase the number of
        // non-zero "rules" in the queue.
        if (nodes_[i].nodes[a].priorities[s] <= 0.0) {
            nodes_[i].nodes[a].priorities[s] = 0.0;
            ++nonZeroPriorities_;
        }
        nodes_[i].nodes[a].priorities[s] += p;
        // We update the maxes if needed
        if (nodes_[i].nodes[a].priorities[s] > nodes_[i].nodes[a].maxV) {
            nodes_[i].nodes[a].maxV = nodes_[i].nodes[a].priorities[s];
            nodes_[i].nodes[a].maxS = s;

            if (nodes_[i].nodes[a].maxV > nodes_[i].maxV) {
                nodes_[i].maxV = nodes_[i].nodes[a].maxV;
                nodes_[i].maxA = a;
            }
        }
    }

    void CPSQueue::reconstruct(State & rets, State & reta) {
        const auto & S = graph_.getS();
        const auto & A = graph_.getA();
        // Initialize retval
        rets = S;
        reta = A;

        // FIXME: find way to avoid calling this all the time when looping over actions/state ids
        const auto partialMatch = [](const Factors & F, const Factors & f, const PartialKeys & keys, size_t id) {
            for (size_t i = 0; i < keys.size(); ++i) {
                const auto key = keys[i];
                const auto val = id % F[key];
                id /= F[key];
                if (f[key] == F[key] || f[key] == val) continue;
                return false;
            }
            return true;
        };

        const auto assignMatch = [](const Factors & F, Factors & f, const PartialKeys & keys, size_t id) {
            for (size_t i = 0; i < keys.size(); ++i) {
                const auto key = keys[i];
                assert(f[key] == F[key] || f[key] == id % F[key]);
                f[key] = id % F[key];
                id /= F[key];
            }
        };

        // First we take the maximum rule to start off. The reason we really
        // want this is that in problems where rewards are relatively sparse,
        // we don't want to risk losing important rules by going around
        // randomly, as it may "fix" the state and action in place with useless
        // rules preventing the actual best stuff from being picked. We only
        // can take the best cheaply, but at least is something.
        double maxIV = -2.0;
        size_t maxI = 0;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i].maxV > maxIV) {
                maxIV = nodes_[i].maxV;
                maxI = i;
            }
        }

        // We want to go over the nodes randomly, but we still want to pick the
        // top. To avoid duplicating the 'canTakeMax' code, we do a slight
        // trick. The first element of order is always equal to S.size(),
        // so there we force the canTakeMax. Otherwise, if the currently
        // iterated element is equal to maxI, then it's the one in the shuffle,
        // and we can skip it (since we already parsed it as first element).
        std::shuffle(std::begin(order_)+1, std::end(order_), rand_);
        for (auto i : order_) {
            if (i == maxI) continue;
            bool canTakeMax;
            // See trick above
            if (i == S.size()) {
                canTakeMax = true;
                i = maxI;
            } else {
                const auto & ps = graph_.getParentSets()[i];
                const auto & node = nodes_[i];
                canTakeMax = partialMatch(A, reta, ps.agents, node.maxA);
                canTakeMax = canTakeMax && partialMatch(S, rets, ps.features[node.maxA], node.nodes[node.maxA].maxS);
            }
            const auto & ps = graph_.getParentSets()[i];
            auto & node = nodes_[i];
            if (canTakeMax) {
                // Add to reta and rets the new values
                assignMatch(A, reta, ps.agents, node.maxA);
                assignMatch(S, rets, ps.features[node.maxA], node.nodes[node.maxA].maxS);

                // Update max of i
                auto & nn = node.nodes[node.maxA];
                if (nn.maxV > 0.0)
                    --nonZeroPriorities_;
                nn.priorities[nn.maxS] = 0.0;
                nn.maxV = nn.priorities.maxCoeff(&nn.maxS);

                node.maxA = 0;
                node.maxV = node.nodes[0].maxV;

                for (size_t a = 1; a < node.nodes.size(); ++a) {
                    if (node.nodes[a].maxV > node.maxV) {
                        node.maxA = a;
                        node.maxV = node.nodes[a].maxV;
                    }
                }
                continue;
            }
            // The default max is not compatible with our current set, so we
            // look around for a half-random, half best choice.
            //
            // Select random compatible action. We know that there is at least
            // one element in its node which is compatible with us (since it
            // contains all possible values of the parents).
            std::shuffle(std::begin(node.order), std::end(node.order), rand_);
            // FIXME: Here we should change 2 things:
            // - First check whether reta or rets are fixed and if so take
            // what's there (optional) (this could be done at the top once for
            // the action, also final assignment at bottom should be done only
            // if not fully set) -> only needed for possible speed.
            // - Secondly, the jj loop should probably span the rest of the
            // block. Here basically we take a shot at the first action that is
            // compatible, and we bail if we don't find anything useful.
            // Instead we should just try again until we have found something,
            // otherwise we might leave a lot of rules behind just because we
            // picked the wrong action to try.
            auto j = 0;
            for (const auto jj : node.order) {
                if (partialMatch(A, reta, ps.agents, jj)) {
                    j = jj;
                    break;
                }
            }

            // Select compatible parent set with highest priority
            const auto & features = ps.features[j];
            auto && nn = node.nodes[j];
            auto x = nn.maxS;
            auto xVal = nn.maxV;
            if (partialMatch(S, rets, features, nn.maxS)) {
                // If the current max is alright...
                // Set to zero
                if (nn.maxV > 0.0)
                    --nonZeroPriorities_;
                nn.priorities[nn.maxS] = 0.0;
                // Find the new max
                nn.maxV = nn.priorities.maxCoeff(&nn.maxS);
                // The new max is <= than the one we popped, and since we are
                // guaranteed not to have picked the global top pick (since
                // we're here looking in the first place) we don't have to
                // propagate the new max above.
            } else {
                // We have to find another one
                xVal = -2.0;

                for (size_t xx = 0; xx < static_cast<size_t>(nn.priorities.size()); ++xx) {
                    if (xx == nn.maxS) continue;
                    if (nn.priorities[xx] > xVal && partialMatch(S, rets, features, xx)) {
                        x = xx;
                        xVal = nn.priorities[x];
                    }
                }
                // Set to zero
                if (xVal > 0.0)
                    --nonZeroPriorities_;
                nn.priorities[x] = 0.0;
                // No max to update tho
            }
            // We only assign if we found something interesting, otherwise we
            // avoid setting variables and wait till the end.
            if (xVal > 0.0) {
                assignMatch(A, reta, ps.agents, j);
                assignMatch(S, rets, features, x);
            }
        }
    }

    double CPSQueue::getNodeMaxPriority(size_t i) const {
        return nodes_[i].maxV;
    }
    unsigned CPSQueue::getNonZeroPriorities() const {
        return nonZeroPriorities_;
    }
}
