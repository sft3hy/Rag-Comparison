"""
Generates a markdown report from evaluation results.
"""

from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader


class ReportGenerator:
    """Generates reports from evaluation artifacts."""

    def __init__(self, templates_dir: str = "src/eval/templates"):
        self.env = Environment(loader=FileSystemLoader(templates_dir))

    def generate_markdown_report(
        self, summary_df: pd.DataFrame, output_path: Path, experiment_name: str
    ):
        """
        Creates a markdown report.

        Args:
            summary_df: DataFrame with summary metrics for each run.
            output_path: Path to save the markdown file.
            experiment_name: Name of the overall experiment.
        """
        template = self.env.get_template("report_template.md.j2")

        # Convert df to markdown table
        summary_table = summary_df.to_markdown()

        report_content = template.render(
            experiment_name=experiment_name,
            summary_table=summary_table,
            plots=[p.name for p in output_path.parent.glob("*.png")],
        )

        with open(output_path, "w") as f:
            f.write(report_content)


# Example template file: src/eval/templates/report_template.md.j2
"""
# Evaluation Report: {{ experiment_name }}

## Summary of Results

This table summarizes the performance of each experimental run.

{{ summary_table }}

## Performance Plots

{% for plot in plots %}
![{{ plot }}](./{{ plot }})
{% endfor %}

"""
