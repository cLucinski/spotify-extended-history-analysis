import datetime
import pandas as pd
from extended_history_analyzer import create_cumulative_listening_line_chart, global_config

def test_cumulative_chart_metadata():
    # Arrange: mock small dataset
    df = pd.DataFrame({
        "ts": [
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z",
            "2025-01-02T12:00:00Z"
        ],
        "ms_played": [3600000, 1800000, 1800000],  # 1 hr, 0.5 hr, 0.5 hr
        "user": ["test_user", "test_user", "test_user"]
    })
    global_config["dark_mode"] = False

    # Act: create chart
    fig = create_cumulative_listening_line_chart(df, group_by="day")

    # Assert: check metadata
    assert "Cumulative Listening History" in fig.layout.title.text
    assert list(fig.data[0].x) == [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2)]
    assert list(fig.data[0].y) == [1.0, 2.0]  # cumulative hours
    assert fig.layout.template.layout.plot_bgcolor == '#E5ECF6'  # light mode background
