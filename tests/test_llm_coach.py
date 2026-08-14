import os
import unittest
from unittest.mock import patch

from app.ipeds_connect.llm_coach import coach_with_optional_llm


class RulesCoachTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_rules_mode_gives_grounded_answer_and_follow_up(self):
        programs = [{
            "cip2_name": "Computer & Information Sciences",
            "awlevel_name": "Bachelor's degree",
            "sunset_label": "Growth/Stable",
            "alignment": "Strong",
            "program_net_pct_change": 5,
            "bls_growth_by_degree": 12,
            "bls_annual_openings_mapped": 1000,
            "job_designations": [],
        }]
        result = coach_with_optional_llm(programs, "I like software")
        self.assertEqual(result["mode"], "rules")
        self.assertIn("Best-fit paths", result["coach_answer"])
        self.assertIn("credential", result["next_question"])


if __name__ == "__main__":
    unittest.main()
