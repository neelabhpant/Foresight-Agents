from crewai import Task

class WalmartTasks:
    def load_and_filter_data_task(self, agent, store_id):
        return Task(
            description=f"Load the dataset from 'data/Walmart_Sales.xlsx', filter it for store {store_id}, save the result to a CSV file in the 'outputs' directory, and output the file path.",
            agent=agent,
            expected_output="The string file path to the new CSV file containing the cleaned data for the specified store."
        )

    def forecast_and_analyze_task(self, agent, context):
        return Task(
            description=(
                "Read the dataset from the file path provided by the previous step. Then, perform a complete sales forecast. "
                "You must train a Prophet model, generate a 52-week forecast, calculate the MAPE, "
                "and extract the regressor coefficients. Your final output must be a dictionary containing all these results."
            ),
            agent=agent,
            context=context,
            expected_output=(
                "A Python dictionary with keys 'mape', 'forecast_total', 'avg_weekly', "
                "'regressor_coefficients', and 'forecast_plot_path'."
            )
        )

    def generate_report_task(self, agent, context, store_id):
        return Task(
            description=(
                "Generate a comprehensive executive summary report in markdown format. "
                f"The report is for store {store_id}. You must use the dictionary of forecasting "
                "results from the previous step to source all metrics and insights. "
                "The report should be well-structured, insightful, and ready for executive review."
            ),
            agent=agent,
            context=context,
            expected_output="A final, complete markdown string containing the full executive summary."
        )