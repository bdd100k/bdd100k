"""Test cases for mot.py."""
import os
import unittest

from .mot import (
    acc_single_video, aggregate_accs, evaluate_single_class,
    METRIC_MAPS, render_results, SUPER_CLASSES)
                 
from .run import read


class TestRenderResults(unittest.TestCase):
    """Test cases for mot render results."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts = read("{}/testcases/track_sample_anns/".format(cur_dir))
    preds = read("{}/testcases/track_predictions.json".format(cur_dir))

    metrics = list(METRIC_MAPS.keys())
    accs = [acc_single_video(gts[0], preds[0])]
    names, accs, items = aggregate_accs(accs)
    summaries = [evaluate_single_class(name, acc)
                 for name, acc in zip(names, accs)]
    eval_results = render_results(summaries, items, metrics)
    
    def test_categories(self) -> None:
        cate_names = ['OVERALL']
        for super_category, categories in SUPER_CLASSES.items():
            cate_names.append(super_category)
            cate_names.extend(categories)

        self.assertEqual(len(self.eval_results), len(cate_names))
        for key in self.eval_results.keys():
            self.assertIn(key, cate_names)

    def test_metrics(self) -> None:
        cate_metrics = list(METRIC_MAPS.values())
        overall_metrics = cate_metrics + ['mIDF1', 'mMOTA', 'mMOTP']

        for cate, metrics in self.eval_results.items():
            if cate == 'OVERALL':
                target_metrics = overall_metrics
            else:
                target_metrics = cate_metrics
            self.assertEqual(len(metrics), len(target_metrics))
            for metric in metrics:
                self.assertIn(metric, target_metrics)


if __name__ == "__main__":
    unittest.main()