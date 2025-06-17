# crew/agents.py
from crewai import Agent

# Import the new function-based tools
from tools.data_tools import load_and_filter_data
from tools.model_tools import forecast_and_analyze_sales
from tools.report_tools import generate_sales_report

class WalmartAgents:
    def data_loader_agent(self, llm):
        return Agent(
            role="Data Ingestion and Preprocessing Expert",
            goal="Load and clean historical Walmart sales data, then filter it for a specific store of interest.",
            backstory=(
                "As a data ingestion expert embedded in a top-tier retail analytics team, "
                "your primary responsibility is to ensure the data pipeline feeds clean, "
                "reliable, and relevant data to the forecasting models. You are meticulous "
                "and have a deep understanding of data quality."
            ),
            tools=[load_and_filter_data], # Use the function directly
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

    def forecasting_agent(self, llm):
        return Agent(
            role="Time Series Modeling Specialist",
            goal=(
                "Generate an accurate 52-week sales forecast for a given store, "
                "evaluate the model's performance using MAPE, and extract the "
                "numerical impact of economic regressors on sales."
            ),
            backstory=(
                "You are a quantitative analyst specializing in time series forecasting. "
                "Trained at a leading financial institution, you excel at using advanced "
                "statistical models like Prophet to predict future trends. Your models "
                "account for seasonality, holidays, and external economic factors."
            ),
            tools=[forecast_and_analyze_sales], # Use the function directly
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

    def reporting_agent(self, llm):
        return Agent(
            role="Senior Retail Business Analyst",
            goal=(
                "Transform raw forecasting outputs into a polished, insightful, and "
                "executive-level markdown report. The report must explain the forecast, "
                "interpret key drivers, and provide actionable business recommendations."
            ),
            backstory=(
                "With years of experience presenting to C-suite executives at a Fortune 500 retailer, "
                "you have a knack for storytelling with data. You can take complex model outputs "
                "and translate them into a clear, concise, and compelling narrative that drives "
                "strategic decision-making."
            ),
            tools=[generate_sales_report], # Use the function directly
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )