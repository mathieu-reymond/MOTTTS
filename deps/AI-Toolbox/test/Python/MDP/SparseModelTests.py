import unittest
import sys
import os
from builtins import range

import pickle
import tempfile

sys.path.append(os.getcwd())
from AIToolbox import MDP

class MDPPythonSparseModelTests(unittest.TestCase):

    def testDefaultBuild(self):
        m1 = MDP.SparseModel(16,4)

        self.assertEqual(m1.getS(), 16)
        self.assertEqual(m1.getA(), 4)
        self.assertEqual(m1.getDiscount(), 1.0)

        for s in range(0, 16):
            for a in range(0, 4):
                for s1 in range(0, 16):
                    if s == s1:
                        self.assertEqual(m1.getTransitionProbability(s,a,s1), 1.0)
                    else:
                        self.assertEqual(m1.getTransitionProbability(s,a,s1), 0.0)
                    self.assertEqual(m1.getExpectedReward(s,a,s1), 0.0)


        m2 = MDP.SparseModel(1,1,0.6)

        self.assertEqual(m2.getS(), 1)
        self.assertEqual(m2.getA(), 1)
        self.assertEqual(m2.getDiscount(), 0.6)

        m2.setDiscount(0.5)
        self.assertEqual(m2.getDiscount(), 0.5)

    def testSetFunctions(self):
        model = MDP.SparseModel(16,4)

        t=[[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0.2,0.8,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0.2,0,0,0,0.8,0,0,0,0,0,0,0,0,0,0],[0.8,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0.2,0.8,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0.2,0,0,0,0.8,0,0,0,0,0,0,0,0,0],[0,0.8,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0.2,0,0,0,0.8,0,0,0,0,0,0,0,0],[0,0,0.8,0.2,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0.8,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0.2,0.8,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0.2,0,0,0,0.8,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0.8,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0.2,0.8,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0.2,0,0,0,0.8,0,0,0,0,0,0],[0,0,0,0,0.8,0.2,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0.8,0,0,0,0.2,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0.2,0.8,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0.2,0,0,0,0.8,0,0,0,0,0],[0,0,0,0,0,0.8,0.2,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,0.8,0,0,0,0.2,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0.2,0,0,0,0.8,0,0,0,0],[0,0,0,0,0,0,0.8,0.2,0,0,0,0,0,0,0,0]],
        [[0,0,0,0,0.8,0,0,0,0.2,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0.2,0.8,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0.2,0,0,0,0.8,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]],
        [[0,0,0,0,0,0.8,0,0,0,0.2,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0.2,0.8,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0.2,0,0,0,0.8,0,0],[0,0,0,0,0,0,0,0,0.8,0.2,0,0,0,0,0,0]],
        [[0,0,0,0,0,0,0.8,0,0,0,0.2,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0.2,0.8,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0.2,0,0,0,0.8,0],[0,0,0,0,0,0,0,0,0,0.8,0.2,0,0,0,0,0]],
        [[0,0,0,0,0,0,0,0.8,0,0,0,0.2,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0.2,0,0,0,0.8],[0,0,0,0,0,0,0,0,0,0,0.8,0.2,0,0,0,0]],
        [[0,0,0,0,0,0,0,0,0.8,0,0,0,0.2,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.8,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]],
        [[0,0,0,0,0,0,0,0,0,0.8,0,0,0,0.2,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.8,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0.8,0.2,0,0]],
        [[0,0,0,0,0,0,0,0,0,0,0.8,0,0,0,0.2,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.8],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0.8,0.2,0]],
        [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]]


        r=[[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0],[0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0],[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0]],
        [[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0]],
        [[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0]],
        [[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0]],
        [[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0]],
        [[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0]],
        [[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0]],
        [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]

        model.setTransitionFunction(t)
        model.setRewardFunction(r)

        for s in range(0, 16):
            for a in range(0, 4):
                # We actually only store SxA in the model, so we need to get
                # the expected reward across next states to do the correct
                # reward check.
                expectedR = sum(v[0] * v[1] for v in zip(r[s][a], t[s][a]))
                for s1 in range(0, 16):
                    self.assertEqual(model.getTransitionProbability(s,a,s1), t[s][a][s1])
                    self.assertEqual(model.getExpectedReward(s,a,s1), expectedR)

    def testPickle(self):
        model = MDP.SparseModel(16,4)

        t=[[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0.2,0.8,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0.2,0,0,0,0.8,0,0,0,0,0,0,0,0,0,0],[0.8,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0.2,0.8,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0.2,0,0,0,0.8,0,0,0,0,0,0,0,0,0],[0,0.8,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0.2,0,0,0,0.8,0,0,0,0,0,0,0,0],[0,0,0.8,0.2,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0.8,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0.2,0.8,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0.2,0,0,0,0.8,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0.8,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0.2,0.8,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0.2,0,0,0,0.8,0,0,0,0,0,0],[0,0,0,0,0.8,0.2,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0.8,0,0,0,0.2,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0.2,0.8,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0.2,0,0,0,0.8,0,0,0,0,0],[0,0,0,0,0,0.8,0.2,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,0.8,0,0,0,0.2,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0.2,0,0,0,0.8,0,0,0,0],[0,0,0,0,0,0,0.8,0.2,0,0,0,0,0,0,0,0]],
        [[0,0,0,0,0.8,0,0,0,0.2,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0.2,0.8,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0.2,0,0,0,0.8,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]],
        [[0,0,0,0,0,0.8,0,0,0,0.2,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0.2,0.8,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0.2,0,0,0,0.8,0,0],[0,0,0,0,0,0,0,0,0.8,0.2,0,0,0,0,0,0]],
        [[0,0,0,0,0,0,0.8,0,0,0,0.2,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0.2,0.8,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0.2,0,0,0,0.8,0],[0,0,0,0,0,0,0,0,0,0.8,0.2,0,0,0,0,0]],
        [[0,0,0,0,0,0,0,0.8,0,0,0,0.2,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0.2,0,0,0,0.8],[0,0,0,0,0,0,0,0,0,0,0.8,0.2,0,0,0,0]],
        [[0,0,0,0,0,0,0,0,0.8,0,0,0,0.2,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.8,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]],
        [[0,0,0,0,0,0,0,0,0,0.8,0,0,0,0.2,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.8,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0.8,0.2,0,0]],
        [[0,0,0,0,0,0,0,0,0,0,0.8,0,0,0,0.2,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.8],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0.8,0.2,0]],
        [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]]


        r=[[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        [[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0],[0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0]],
        [[0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0],[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0]],
        [[0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0]],
        [[0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0]],
        [[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0]],
        [[0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0]],
        [[0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0]],
        [[0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0]],
        [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]

        model.setTransitionFunction(t)
        model.setRewardFunction(r)
        model.setDiscount(0.4)

        with tempfile.TemporaryFile() as fp:
            pickle.dump(model, fp)
            fp.seek(0)
            newModel = pickle.load(fp)

        self.assertEqual(model.getS(),        newModel.getS())
        self.assertEqual(model.getA(),        newModel.getA())
        self.assertEqual(model.getDiscount(), newModel.getDiscount())

        for s in range(model.getS()):
            for a in range(model.getA()):
                for s1 in range(model.getS()):
                    self.assertAlmostEqual(
                        model.getTransitionProbability(s,a,s1),
                        newModel.getTransitionProbability(s,a,s1)
                    )
                    self.assertAlmostEqual(
                        model.getExpectedReward(s,a,s1),
                        newModel.getExpectedReward(s,a,s1)
                    )

if __name__ == '__main__':
    unittest.main(verbosity=2)

