# crew/main.py
import yaml
from crewai import Crew, Process
from langchain_openai import ChatOpenAI

from .agents import WalmartAgents
from .tasks import WalmartTasks

class WalmartForecastingCrew:
    def __init__(self, store_id: int):
        self.store_id = store_id
        self.agents = WalmartAgents()
        self.tasks = WalmartTasks()
        self._load_config_and_llm()

    def _load_config_and_llm(self):
        """Loads config and initializes the LLM."""
        try:
            with open('config/models.yaml', 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            config = {'reporting_llm': {'model_name': 'gpt-4o', 'temperature': 0.4}}

        llm_config = config.get('reporting_llm', {})
        self.llm = ChatOpenAI(
            model_name=llm_config.get('model_name', 'gpt-4o'),
            temperature=llm_config.get('temperature', 0.4)
        )

    def run(self):
        data_loader = self.agents.data_loader_agent(llm=self.llm)
        forecaster = self.agents.forecasting_agent(llm=self.llm)
        reporter = self.agents.reporting_agent(llm=self.llm)

        # Create tasks
        load_task = self.tasks.load_and_filter_data_task(data_loader, self.store_id)
        
        # FIX: Pass the context list directly
        forecast_task = self.tasks.forecast_and_analyze_task(
            agent=forecaster, 
            context=[load_task]
        )
        
        # FIX: Pass the context list directly
        report_task = self.tasks.generate_report_task(
            agent=reporter, 
            context=[forecast_task], 
            store_id=self.store_id
        )

        # Assemble the crew
        crew = Crew(
            agents=[data_loader, forecaster, reporter],
            tasks=[load_task, forecast_task, report_task],
            process=Process.sequential,
            verbose=True,
            memory=False
        )

        result = crew.kickoff()
        return result

if __name__ == "__main__":
    store_to_forecast = 1
    forecasting_crew = WalmartForecastingCrew(store_id=store_to_forecast)
    final_report = forecasting_crew.run()
    
    print("\n\n################################")
    print("## Final Executive Report ##")
    print("################################\n")
    print(final_report)