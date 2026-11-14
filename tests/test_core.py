import unittest, tempfile
import main
class TestCore(unittest.TestCase):
    def test_import(self):
        self.assertTrue(hasattr(main,"__name__"))
    def test_training_and_inference(self):
        with tempfile.TemporaryDirectory() as d:
            m=main.train(d)
            self.assertGreater(m['accuracy'],0.8)
            from sklearn.datasets import load_breast_cancer
            probs=main.predict(load_breast_cancer().data[:2],d)
            self.assertEqual(len(probs),2)
            self.assertTrue(all(0<=p<=1 for p in probs))
    def test_drift(self):
        with tempfile.TemporaryDirectory() as d:
            main.train(d)
            from sklearn.datasets import load_breast_cancer
            self.assertIn('features_over_2sd',main.drift(load_breast_cancer().data[:10],d))

if __name__=="__main__":unittest.main()
