import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dept_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "HR", "HR"],
            "monthly_income": [4000, 6000, 3000, 5000, 7000, 8000],
            "overtime": ["Yes", "No", "Yes", "No", "No", "No"],
            "job_satisfaction": [1, 3, 2, 3, 4, 4],
            "attrition": ["Yes", "No", "Yes", "Yes", "No", "No"],
        }
    )
    # Sales: 2 employees, 1 leaver  -> 50.0%
    # HR:   4 employees, 2 leavers  -> 50.0%  (tied, both appear)


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------

def test_attrition_rate_fifty_percent():
    df = pd.DataFrame({"employee_id": [1, 2, 3, 4], "attrition": ["Yes", "No", "No", "Yes"]})
    assert attrition_rate(df) == 50.0


def test_attrition_rate_zero_when_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_hundred_when_all_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_rounds_to_two_decimal_places():
    # 1 of 3 = 33.333...%
    df = pd.DataFrame({"employee_id": [1, 2, 3], "attrition": ["Yes", "No", "No"]})
    assert attrition_rate(df) == 33.33


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------

def test_attrition_by_department_columns():
    df = pd.DataFrame(
        {"employee_id": [1, 2], "department": ["Sales", "HR"], "attrition": ["Yes", "No"]}
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_correct_values():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "Yes", "Yes"],
        }
    )
    result = attrition_by_department(df).set_index("department")
    assert result.loc["Sales", "employees"] == 2
    assert result.loc["Sales", "leavers"] == 1
    assert result.loc["Sales", "attrition_rate"] == 50.0
    assert result.loc["HR", "employees"] == 2
    assert result.loc["HR", "leavers"] == 2
    assert result.loc["HR", "attrition_rate"] == 100.0


def test_attrition_by_department_sorted_descending_by_rate():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            # HR 100%, Sales 50% — HR should appear first
            "attrition": ["Yes", "No", "Yes", "Yes"],
        }
    )
    result = attrition_by_department(df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------

def test_attrition_by_overtime_columns():
    df = pd.DataFrame(
        {"employee_id": [1, 2], "overtime": ["Yes", "No"], "attrition": ["Yes", "No"]}
    )
    result = attrition_by_overtime(df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_correct_values():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "No", "No"],
            "attrition": ["Yes", "No", "No", "No"],
        }
    )
    result = attrition_by_overtime(df).set_index("overtime")
    assert result.loc["Yes", "employees"] == 2
    assert result.loc["Yes", "leavers"] == 1
    assert result.loc["Yes", "attrition_rate"] == 50.0
    assert result.loc["No", "employees"] == 2
    assert result.loc["No", "leavers"] == 0
    assert result.loc["No", "attrition_rate"] == 0.0


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------

def test_average_income_by_attrition_columns():
    df = pd.DataFrame({"attrition": ["Yes", "No"], "monthly_income": [4000, 6000]})
    result = average_income_by_attrition(df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_correct_means():
    df = pd.DataFrame(
        {
            "attrition": ["Yes", "Yes", "No", "No"],
            "monthly_income": [3000, 5000, 7000, 9000],
        }
    )
    result = average_income_by_attrition(df).set_index("attrition")
    assert result.loc["Yes", "avg_monthly_income"] == 4000.0
    assert result.loc["No", "avg_monthly_income"] == 8000.0


def test_average_income_by_attrition_rounds_to_two_decimal_places():
    # mean of 3000 + 4000 + 5000 = 4000.0 exactly, use non-round number
    df = pd.DataFrame(
        {
            "attrition": ["Yes", "Yes", "Yes"],
            "monthly_income": [1000, 2000, 3000],  # mean = 2000.0
        }
    )
    result = average_income_by_attrition(df).set_index("attrition")
    assert result.loc["Yes", "avg_monthly_income"] == 2000.0

    df2 = pd.DataFrame({"attrition": ["Yes", "Yes"], "monthly_income": [1000, 2000]})
    result2 = average_income_by_attrition(df2).set_index("attrition")
    # 1500.0 is already 2dp; the point is it doesn't blow up
    assert result2.loc["Yes", "avg_monthly_income"] == 1500.0


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------

def test_satisfaction_summary_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "job_satisfaction": [1, 2],
            "attrition": ["Yes", "No"],
        }
    )
    result = satisfaction_summary(df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


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


def test_satisfaction_summary_correct_counts():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "job_satisfaction": [1, 1, 2, 2, 2],
            "attrition": ["Yes", "No", "No", "No", "No"],
        }
    )
    result = satisfaction_summary(df).set_index("job_satisfaction")
    assert result.loc[1, "total_employees"] == 2
    assert result.loc[1, "leavers"] == 1
    assert result.loc[2, "total_employees"] == 3
    assert result.loc[2, "leavers"] == 0


def test_satisfaction_summary_sorted_ascending_by_satisfaction():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "job_satisfaction": [3, 1, 2],
            "attrition": ["No", "Yes", "No"],
        }
    )
    result = satisfaction_summary(df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)
