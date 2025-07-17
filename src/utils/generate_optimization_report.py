import pandas as pd
import plotly.express as px

def generate_report(results_path="src/optuna_results.csv", output_path="optimization_report.html", trail_start=None, trail_stop=None):
    """
    Generates a parallel coordinates plot from Optuna results to visualize hyperparameter optimization.

    Args:
        results_path (str): The path to the Optuna results CSV file.
        output_path (str): The path to save the generated HTML report.
    """
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Error: The file '{results_path}' was not found.")
        return

    # Filter trials based on the specified range
    if trail_start is not None:
        df = df[df['number'] >= trail_start]
    if trail_stop is not None:
        df = df[df['number'] <= trail_stop]

    # Calculate gamma
    if 'params_alpha' in df.columns and 'params_beta' in df.columns:
        df['params_gamma'] = 1 - df['params_alpha'] - df['params_beta']
    else:
        print("Error: 'params_alpha' or 'params_beta' not found in the results file.")
        return

    # Create the parallel coordinates plot
    fig = px.parallel_coordinates(
        df,
        dimensions=['params_alpha', 'params_beta', 'params_gamma', 'value'],
        color="value",
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={
            "params_alpha": "Alpha",
            "params_beta": "Beta",
            "params_gamma": "Gamma",
            "value": "Composite Score",
        },
        title="Bayesian Optimization of Composite Score Weights",
    )

    # Save the plot to an HTML file
    fig.write_html(output_path)
    print(f"Optimization report saved to '{output_path}'")

if __name__ == "__main__":
    # Example usage:
    # Generate a report for all trials
    generate_report()

    # Generate a report for trials from 10 to 50
    generate_report(trail_start=10, trail_stop=50, output_path="optimization_report_10_50.html")

    # Generate a report for trials from 50 onwards
    generate_report(trail_start=50, output_path="optimization_report_50_onwards.html")
