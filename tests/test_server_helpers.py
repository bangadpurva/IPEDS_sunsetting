import unittest

from app.ipeds_connect.server import match_institutions


class InstitutionMatchingTests(unittest.TestCase):
    def test_location_and_budget_are_constraints(self):
        institutions = [
            {"institution_name": "Detroit College", "city": "Detroit", "state": "MI", "average_net_price": 15000},
            {"institution_name": "Detroit Premium", "city": "Detroit", "state": "MI", "average_net_price": 35000},
            {"institution_name": "Lansing College", "city": "Lansing", "state": "MI", "average_net_price": 12000},
        ]
        result = match_institutions(institutions, {"location": "Detroit", "max_annual_cost": 20000})
        self.assertEqual([row["institution_name"] for row in result], ["Detroit College"])


if __name__ == "__main__":
    unittest.main()
