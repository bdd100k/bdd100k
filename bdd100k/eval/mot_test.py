"""Test cases for mot.py."""
import os
import unittest

from scalabel.eval.mot import (
    METRIC_MAPS,
    acc_single_video_mot,
    aggregate_accs,
    evaluate_single_class,
    evaluate_track,
    render_results,
)
from scalabel.label.io import group_and_sort, load

from bdd100k.common.utils import DEFAULT_COCO_CONFIG

from .mot import CLASSES, IGNORE_CLASSES, SUPER_CLASSES


class TestBDD100KMotEval(unittest.TestCase):
    """Test cases for BDD100K MOT evaluation."""

    def test_mot(self) -> None:
        """Check mot evaluation correctness."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gts = group_and_sort(
            load("{}/testcases/track_sample_anns.json".format(cur_dir))
        )
        preds = group_and_sort(
            load("{}/testcases/track_predictions.json".format(cur_dir))
        )
        result = evaluate_track(
            acc_single_video_mot, gts, preds, DEFAULT_COCO_CONFIG
        )
        overall_reference = {
            "IDF1": 0.7101073676416142,
            "MOTA": 0.6420070762302992,
            "MOTP": 0.871614396957838,
            "FP": 126,
            "FN": 942,
            "IDSw": 45,
            "MT": 62,
            "PT": 47,
            "ML": 33,
            "FM": 66,
            "mIDF1": 0.32247819436558384,
            "mMOTA": 0.24324204637536687,
            "mMOTP": 0.5001285135514636,
        }
        for key in result["OVERALL"]:
            self.assertAlmostEqual(
                result["OVERALL"][key], overall_reference[key]
            )


class TestRenderResults(unittest.TestCase):
    """Test cases for mot render results."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts = load("{}/testcases/track_sample_anns.json".format(cur_dir))
    preds = load("{}/testcases/track_predictions.json".format(cur_dir))

    metrics = list(METRIC_MAPS.keys())
    accs = [acc_single_video_mot(gts, preds, CLASSES, IGNORE_CLASSES)]
    names, accs, items = aggregate_accs(accs, CLASSES, SUPER_CLASSES)
    summaries = [
        evaluate_single_class(name, acc) for name, acc in zip(names, accs)
    ]
    eval_results = render_results(
        summaries, items, metrics, CLASSES, SUPER_CLASSES
    )

    def test_categories(self) -> None:
        """Check the correctness of the 1st-level keys in eval_results."""
        cate_names = ["OVERALL"]
        for super_category, categories in SUPER_CLASSES.items():
            cate_names.append(super_category)
            cate_names.extend(categories)

        self.assertEqual(len(self.eval_results), len(cate_names))
        for key in self.eval_results:
            self.assertIn(key, cate_names)

    def test_metrics(self) -> None:
        """Check the correctness of the 2nd-level keys in eval_results."""
        cate_metrics = list(METRIC_MAPS.values())
        overall_metrics = cate_metrics + ["mIDF1", "mMOTA", "mMOTP"]

        for cate, metrics in self.eval_results.items():
            if cate == "OVERALL":
                target_metrics = overall_metrics
            else:
                target_metrics = cate_metrics
            self.assertEqual(len(metrics), len(target_metrics))
            for metric in metrics:
                self.assertIn(metric, target_metrics)


if __name__ == "__main__":
    unittest.main()
