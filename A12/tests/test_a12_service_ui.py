from A12.service.ui import run_a12_tab


def test_run_a12_tab_returns_structured_error_for_missing_file():
    result, summary = run_a12_tab(None, "B")
    assert result["status"] == "error"
    assert "message" in result
    assert "Prediction failed" in summary
