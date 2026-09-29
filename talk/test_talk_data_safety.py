"""The talk must never embed challenge data: only a synthetic cohort and our own aggregate scores."""
import builtins
import re
import sys
import unittest
from pathlib import Path

TALK = Path(__file__).resolve().parent
ROOT = TALK.parent
# guard.py necessarily names the forbidden files, and the README states what is excluded.
SCANNED = [p for p in TALK.rglob("*") if p.is_file() and p.suffix in {".py", ".csv", ".ipynb", ".html"}
           and p.name not in {"guard.py", Path(__file__).name}]
PATTERN = re.compile(r"DB-1773398340961|submission_|oof_predictions|reference_archive|/artifacts/")


class TalkDataSafetyTest(unittest.TestCase):
    def test_sources_never_name_challenge_files(self):
        for path in SCANNED:
            self.assertIsNone(PATTERN.search(path.read_text(errors="ignore")), path)

    def test_score_tables_hold_aggregates_only(self):
        allowed = {"cv_scores.csv": "stage,model,c_hep,c_death,weighted",
                   "leaderboard.csv": "submission,public_score",
                   "cohort_stats.csv": "endpoint,n_patients,events"}
        for path in (TALK / "data").glob("*.csv"):
            self.assertEqual(path.read_text().splitlines()[0], allowed.get(path.name), path)

    def test_guard_refuses_challenge_files(self):
        sys.path.insert(0, str(TALK))
        import guard
        original = builtins.open
        try:
            guard.install()
            with self.assertRaises(PermissionError):
                open(ROOT / "data" / "DB-1773398340961.csv")
            with self.assertRaises(PermissionError):
                open(ROOT / "reference_archive" / "notes.txt")
            with open(TALK / "guard.py") as handle:      # our own files stay readable
                self.assertTrue(handle.readline())
        finally:
            builtins.open = original


class GateMathTest(unittest.TestCase):
    """The false-positive gate formulas in talk/gates.py."""

    def setUp(self):
        sys.path.insert(0, str(TALK))
        import gates
        self.G = gates

    def test_blend_weights(self):
        G = self.G
        self.assertAlmostEqual(G.optimal_value_weight(1.0, 1.0, 0.0), 0.5)
        v = G.optimal_value_weight(1.0, 2.0, 1.0)             # perfectly shared errors
        self.assertAlmostEqual(G.blend_sigma(1.0, 2.0, 1.0, v), 1.0)   # no gain possible
        self.assertLess(G.blend_sigma(1.0, 2.0, 0.0, G.optimal_value_weight(1.0, 2.0, 0.0)), 1.0)
        w = G.value_to_rank_weight(0.2, 0.6, 1.4)
        self.assertAlmostEqual(G.rank_to_value_weight(w, 0.6, 1.4), 0.2)

    def test_noise_bars(self):
        G = self.G
        self.assertLess(G.null_max_gain(0.003, 5), G.null_max_gain(0.003, 272))
        self.assertLess(G.hanley_mcneil_se(0.9, 40, 400), G.hanley_mcneil_se(0.9, 10, 100))
        self.assertEqual(G.verdict(0.001, 0.003, 5, 0.01, True), "within noise")
        self.assertEqual(G.verdict(0.05, 0.003, 5, 0.01, True), "implausible: no mechanism")
        self.assertEqual(G.verdict(0.009, 0.003, 5, 0.01, True), "credible")


if __name__ == "__main__":
    unittest.main()
