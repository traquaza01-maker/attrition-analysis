import pandas as pd
from metrics import attrition_rate, attrition_by_department, satisfaction_summary


def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rate_is_per_group_not_share_of_leavers():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [1, 1, 2, 2],
            "attrition": ["Yes", "No", "Yes", "Yes"],
        }
    )
    result = satisfaction_summary(df).set_index("job_satisfaction")
    # group 1: 1 of 2 left = 50.0%, not 33.33% (share of 3 total leavers)
    assert result.loc[1, "attrition_rate"] == 50.0
    # group 2: 2 of 2 left = 100.0%, not 66.67%
    assert result.loc[2, "attrition_rate"] == 100.0
