# tools/report_tools.py
import os
import yaml
from crewai.tools import tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

@tool("GPT-powered Executive Report Generator")
def generate_sales_report(store_id: int, mape: float, forecast_total: int, avg_weekly: int, regressor_coefficients: str, forecast_plot_path: str) -> str:
    """
    Uses a powerful LLM to generate an executive-level markdown report
    summarizing sales forecasts, model accuracy, and business insights.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "Error: OPENAI_API_KEY environment variable not set."

    try:
        with open('config/models.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception:
        config = {'reporting_llm': {'model_name': 'gpt-4o', 'temperature': 0.4}}

    llm_config = config.get('reporting_llm', {})
    model_name = llm_config.get('model_name', 'gpt-4o')
    temperature = llm_config.get('temperature', 0.4)

    prompt_template = PromptTemplate.from_template("""
You are a senior retail sales analyst generating a forecast briefing for executives. Based on the data below, write a compelling, insightful report for **Store {store_id}**.

---

### Model Accuracy

- **MAPE (Mean Absolute Percentage Error)**: {mape}%
- *This measures the average percentage error of the sales forecast model for this store. A lower value signifies a more accurate model.*

### 52-Week Forecast Summary

- **Total Forecasted Sales**: ${forecast_total:,}
- **Average Weekly Sales**: ${avg_weekly:,}
- *This forecast is based on a Prophet model trained with historical sales, U.S. holidays, and key macroeconomic signals.*

### Top Drivers of Sales (Regressor Impact)

{coefs}

**Analysis of Drivers:**
- Explain which factors most positively or negatively affect sales based on the coefficient values.
- Why might these factors be particularly important for retail performance?
- Highlight any surprising or counterintuitive signals from the data.

### Business Insight & Recommendations

Based on the forecast and the key drivers, generate a concise yet powerful business summary:
- What is the overall sales trajectory for the next year?
- Should the team prepare for growth, a slowdown, or stable performance?
- Provide **two actionable recommendations** based on the insights (e.g., related to inventory planning during high CPI periods, or marketing adjustments based on fuel price trends).

Be direct, data-driven, and write in an executive-ready markdown format.
Avoid generic phrasing — tailor insights to the specific signals provided.
""")
    llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=openai_api_key)
    chain = LLMChain(prompt=prompt_template, llm=llm, output_parser=StrOutputParser())
    return chain.invoke({
        "store_id": store_id,
        "mape": mape,
        "forecast_total": forecast_total,
        "avg_weekly": avg_weekly,
        "coefs": regressor_coefficients
    })