from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from crewai import Agent, Task, Crew, LLM
import json
import os
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_cors import CORS


load_dotenv()

serper_api_key = os.getenv("SERPER_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": ["Content-Type"]}})
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7,google_api_key=gemini_api_key)


class AIStartupConsultant:
    def __init__(self, llm, mode="full"):
        self.llm = LLM(model="gemini/gemini-2.0-flash-exp", temperature=0.7)
        self.mode = mode
        self.crew = None

    def create_agent(self, name, role, goal, backstory):
        """Creates an individual agent."""
        return Agent(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            llm=self.llm,
        )

    def create_task(self, agent):
        """Creates an individual task for an agent."""
        return Task(
            description=agent.goal,
            expected_output=f"A detailed report from {agent.role}.",
            agent=agent,
        )

    def create_agents(self, sector):
        """Creates agents dynamically based on the selected mode."""
        all_agents = {
            "industry_research": self.create_agent(
                f"{sector} Industry Analyst", "Industry Researcher",
                f"Gather insights about the {sector} industry trends, opportunities, and challenges.",
                "An expert in industry analysis and startup opportunities."
            ),
            "market_research": self.create_agent(
                "Market Researcher", "Market Analyst",
                f"Analyze the {sector} market size, value, and key competitors.",
                "A specialist in market trends and competitor analysis."
            ),
            "cost_estimator": self.create_agent(
                "Financial Estimator", "Finance & Investment Expert",
                f"Estimate startup costs, funding requirements, and potential revenue in {sector}.",
                "A finance expert who helps startups plan their financials."
            ),
            "legal_compliance": self.create_agent(
                "Legal Compliance Advisor", "Legal Consultant",
                f"Provide legal structure recommendations and compliance requirements for {sector} startups.",
                "A legal advisor specializing in startup regulations."
            ),
            "tech_consultant": self.create_agent(
                "Technology Consultant", "Tech Expert",
                f"Suggest trending technologies and AI implementations for {sector} startups.",
                "A tech-savvy consultant with expertise in emerging technologies."
            )
        }

        if self.mode == "basic":
            return {key: all_agents[key] for key in ["industry_research", "market_research", "cost_estimator", "legal_compliance", "tech_consultant"]}
        
        additional_agents = {
            "innovation_expert": self.create_agent(
                "Innovation Strategist", "Startup Innovator",
                f"Identify gaps and innovation opportunities in {sector} to ensure startup success.",
                "A strategic thinker who finds unique angles for startup growth."
            ),
            "loophole_finder": self.create_agent(
                "Industry Loophole Finder", "Strategic Advisor",
                f"Identify gaps and loopholes in the {sector} industry that a startup can capitalize on.",
                "A business strategist who finds inefficiencies in industries to exploit."
            ),
            "risk_analyst": self.create_agent(
                "Risk Analyst", "Risk & Security Consultant",
                f"Identify financial, operational, and legal risks in the {sector} startup model.",
                "A risk management specialist with experience in minimizing business threats."
            ),
        }
        return {**all_agents, **additional_agents}

    def create_tasks(self, agents):
        """Creates tasks dynamically for each agent."""
        return {key: self.create_task(agent) for key, agent in agents.items()}

    def extract_sector(self, user_input):
        """Extracts the startup sector from user input."""
        extraction_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""
            Extract the sector or field of startup from the user_input :
            "{user_input}"

            Respond in JSON format:
            {{"user_input": "..."}}
            And Do not Respond with any extra text instead this
            """
        )

        extraction_chain = LLMChain(prompt=extraction_prompt, llm=llm)
        unprocessed_extracted_data = extraction_chain.run(user_input)
        extracted_data = json.loads(str(unprocessed_extracted_data))
        return extracted_data["user_input"]
    
    def process_startup_request(self, user_query):
        """Processes user requests and generates startup guidance."""
        sector = self.extract_sector(user_query)
        if sector == "null":
            return "Please specify the sector for your startup (e.g., EdTech, FinTech, HealthTech, etc.)."

        agents = self.create_agents(sector)
        tasks = self.create_tasks(agents)
        self.crew = Crew(agents=list(agents.values()), tasks=list(tasks.values()))
        response = self.crew.kickoff(inputs={"query": user_query})
        return response, sector
    
@app.route('/consult', methods=['POST'])
def generate_roadmap():
    """Endpoint to generate a response based on query and type."""
    data = request.get_json()

    if not data or 'query' not in data or 'type' not in data:
        return jsonify({"error": "Both 'query' and 'type' fields are required in the JSON body"}), 400

    user_input = data['query']
    startup_consult_type = data['type']

    system = AIStartupConsultant(llm, startup_consult_type)
    try:
        response, sector = system.process_startup_request(user_input)
        response = str(response)
        return jsonify({"result": response, "sector": sector})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(
        host=os.getenv('HOST', '0.0.0.0'), 
        port=int(os.getenv('PORT', '8080')),
        debug=False
    )


