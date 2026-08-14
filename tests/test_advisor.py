import unittest

from app.ipeds_connect.advisor import StudentProfile, agentic_recommend, degree_matches, profile_from_prompt, recommend


def program(field, award, jobs=None, growth=8):
    return {
        "cip2_name": field,
        "awlevel_name": award,
        "sunset_label": "Growth/Stable",
        "alignment": "Strong",
        "program_net_pct_change": 5,
        "bls_growth_by_degree": growth,
        "bls_annual_openings_mapped": 10000,
        "job_designations": jobs or [],
    }


class AdvisorTests(unittest.TestCase):
    def test_explicit_degree_is_a_hard_constraint(self):
        programs = [
            program("Computer & Information Sciences", "Master's degree", growth=30),
            program("Computer & Information Sciences", "Award < Bachelor's", growth=40),
            program("Mathematics & Statistics", "Bachelor's degree", growth=5),
        ]
        results = recommend(programs, StudentProfile(("technology",), degree_level="bachelor"))
        self.assertEqual([row["awlevel_name"] for row in results], ["Bachelor's degree"])

    def test_professional_degree_matches_doctoral_not_bachelor(self):
        row = program("Health Professions", "First professional degree")
        self.assertTrue(degree_matches(row, "doctoral"))
        self.assertFalse(degree_matches(row, "bachelor"))

    def test_prompt_extracts_location_and_budget(self):
        profile, _ = profile_from_prompt("I want a bachelor's program near Detroit, under $20,000")
        self.assertEqual(profile.degree_level, "bachelor")
        self.assertEqual(profile.location, "Detroit")
        self.assertEqual(profile.max_annual_cost, 20000)

    def test_in_demand_is_not_treated_as_a_location(self):
        profile, _ = profile_from_prompt("What jobs are in demand?")
        self.assertIsNone(profile.location)

    def test_healthcare_prompt_excludes_unrelated_data_job(self):
        programs = [
            program(
                "Health Professions",
                "Bachelor's degree",
                jobs=[
                    {"title": "Data scientists", "projected_growth": 30, "annual_openings": 20000},
                    {"title": "Medical assistants", "projected_growth": 15, "annual_openings": 100000},
                ],
            )
        ]
        result = agentic_recommend(programs, "I want a bachelor's healthcare career")
        self.assertEqual([job["title"] for job in result["job_designations"]], ["Medical assistants"])


if __name__ == "__main__":
    unittest.main()
