import unittest
from scorer import KeywordScorer

class TestKeywordScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = KeywordScorer()
        
    def test_positive_signal(self):
        text = "In summary, the key idea of quantum computing is entanglement."
        score = self.scorer(text)
        self.assertGreater(score, 1.0)
        
    def test_negative_signal(self):
        text = "I don't know. As an AI I cannot provide that."
        score = self.scorer(text)
        self.assertLess(score, 0.0)
        
    def test_length_effect(self):
        short = "Quantum computing is interesting."
        long = "Quantum computing is a revolutionary approach..." * 10
        self.assertGreater(self.scorer(long), self.scorer(short))
        
if __name__ == '__main__':
    unittest.main()