from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
import os
from pathlib import Path

from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage
import pandas as pd
import numpy as np

from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI, OpenAI

from typing import Literal


# Extending state definition for a multi-agent workflow in LangGraph.
class AgentState(MessagesState):
    # The 'next' field indicates where to route to next
    next: str


df = pd.read_csv('data_extract.csv')# Reading CSV file
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

stats_agent = create_pandas_dataframe_agent(#Using Daaframe agent from toolkit
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True
)



members = ["stats_agent", "analyst_agent", "file_checker_agent", "communicator_agent"]#List of all the working agents
options = members + ["END"]
system_prompt = (f"""
        You are a supervisor tasked with managing a conversation between the following workers: {members}. 
        Given the following user request, respond with the worker to act next. 
        Each worker will perform a task and respond with their results and status, read through the response and look for what needs to be done next.
        Check if the files exist before starting, if the files don't exist trigger the necessary agent.
        When all tasks are complete, respond with END.
        """
)
#Router state definition used to control conditional routing between agents in a LangGraph-based multi-agent workflow
class Router(TypedDict):
    # Worker to rout to next. If no workers needed, route to END
    next: Literal[(*options,)]


def supervisor_node(state: AgentState) -> AgentState:#Defining Supervisor Agent with system prompt
    messages = [
        f"""{system_prompt} {state["messages"][-1]}
        """
        ]
    response = llm.with_structured_output(Router).invoke(messages)
    print(messages)
    next_ = response["next"]
    if next_ == "END":
        next_ = END
    return {"next": next_}


def stats_agent_node(state: AgentState):#Defining Supervisor Agent with prompt engineering
    # message = state["messages"][-1].content
    prompts = [
            f"""
                You are a data cleaner prepping for analysis, you will be given data in the form of a CSV. 
                Let's think step by step. First work out the schema of the data you receive. 
                Then compare the data you have to the schema you determined. 
                Don't move on until you have figured out the schema.
                Here is your dataset: data_extract.csv
                Follow these steps:
                1. Convert the column Date into Date datatype
                2. Add these columns to the dataset:
                    month:  YYYY-MM
                    week: date range for the week
                3. Retrieve today's date
                4. Filter your data on the date range for the first day of the month that is six months ago from today's date through the end of last month.
                5. Replace any missing numeric values with 0 and alphanumeric values with 'N/A'
                6. Write the new dataset with the added columns to data_dump/stats/stats.csv
            """,
            f"""
                You are a statistician prepping for analysis, you will be given data in the form of a CSV. 
                Let's think step by step. First work out the schema of the data you receive. 
                Then compare the data you have to the schema you determined. 
                Don't move on until you have figured out the schema.
                Here is your dataset: data_dump/stats/stats.csv
                Check if the columns have been normalized, if they are already normalized move on to the next task
                Otherwise, normalize the existing column names so they are all lowercase and replace spaces with underscores then update data_dump/stats/stats.csv with the new column names.
                You will perform a series of calculations each written to a new file:
                    1. Calculate the daily average for each metric grouped by month, creative vertical, media type, and site. Write the new dataset including only the month, creative vertical, media type, site and each aggregation to data_dump/stats/stats_monthly.csv with renamed metrics.
                    2. Calculate the daily average for each metric grouped by month. Exclude any missing values from the mean. Write the new dataset including only the month and daily averages for each metric to data_dump/stats/stats_monthly_avg.csv.
            """,
            f"""
                You are a statistician prepping for analysis, you will be given data in the form of a CSV. 
                Let's think step by step. First work out the schema of the data you receive. 
                Then compare the data you have to the schema you determined. 
                Don't move on until you have figured out the schema.
                Here is your dataset: data_dump/stats/stats.csv
                Check if the columns have been normalized, if they are already normalized move on to the next task
                Otherwise, normalize the existing column names so they are all lowercase and replace spaces with underscores then update data_dump/stats/stats.csv with the new column names.
                You will perform a series of calculations written to a new file:
                    1. Take the maximum daily average from data_dump/stats/stats_monthly.csv for each metric grouped by creative vertical, media type, and site. Write the new dataset including only the creative vertical, media type, site and the maximum daily average for each metric to data_dump/stats/stats_thresholds.csv.
                    2. Take the minimum daily average from data_dump/stats/stats_monthly.csv for each metric grouped by creative vertical, media type, and site. Add the new columns for each metric to data_dump/stats/stats_thresholds.csv.
                    3. Calculate the upper and lower thresholds for each metric with the upper threshold being 200% of the maxmimum daily average and the lower threshold being 50% of the minimum daily average from data_dump/stats/stats_thresholds.csv. Add the new columns to data_dump/stats/stats_thresholds.csv.
                    4. Calculate for each metric in data_dump/stats/stats_monthly.csv a flag that is True if there are less than 3 non-zero values and False otherwise, make sure to include "new_data_" in these new column names then add them to data_dump/stats/stats_thresholds.csv.
                    5. Calculate the daily average for each metric grouped by creative vertical, media type and site for the first two weeks of November 2024, ignore any missing values. Add the new columns to data_dump/stats/stats_thresholds.csv with "two_weeks_avg_" included in the new column names.
                """]
    tasks = [stats_agent.invoke(prompt) for prompt in prompts]
    return {# returning output of task to the State Object
        "messages": [
            HumanMessage(content=tasks[-1]['output'], name="stats_agent"),
        ]
    }


def analyst_agent_node(state: AgentState):#Defining analyst Agent with prompt engineering
    # message = state["messages"][-1].content
    prompts = [f"""
        You are a data engineer tasked with making sure all data has been loaded properly.
        You will receive data in the form of a CSV.
        First determine the schema of the data you received.
        Do not move on until you have determined the schema and verified the data fits into the schema.
        Ignore any missing values.
        Calculate the daily average of each metric for the first two weeks of November 2024.
        Here is your dataset: data_dump/stats/stats.csv.
        Compare it to the daily averages in this dataset: data_dump/stats/stats_monthly_avg.csv.
        Focus on finding metrics with daily averages that are greater than 200% of the daily average from the monthly dataset or 50% less than the daily average from the monthly dataset.
        Write the metrics with their value, the comparison threshold and their values to data_dump/analysis/anomaly_report.csv.
        """,
        f"""
        Does data_dump/analysis/anomaly_report.csv exist?
        If so, read the file and proceed with the next task, if not stop here and report it is missing.
        The names of the columns that contain metrics you need to check are in the "metric" column of data_dump/analysis/anomaly_report.csv.
        Once you've compiled a list of the metrics you need to check, you will check a file with thresholds and the daily average for a two week period.
        Check that data_dump/stats/stats_thresholds.csv contains columns with "two_weeks_avg_" in the name and the data is valid.
        If the columns don't exist, use the data from data_dump/stats/stats.csv to calculate the daily average for each metric grouped by creative vertical, media type, and site for the first two weeks of November 2024, ignore any missing values.
        Compare the averages for those metrics in for two_weeks_avg to the upper_threshold and lower_threshold averages.
        Here is your data: data_dump/stats/stats_thresholds.csv.
        Focus on finding which daily averages from the two weeks that are outside the upper and lower thresholds.
        Write the metric, creative vertical, media type, site, comparison threshold and their values to data_dump/analysis/anomaly_report_detailed.csv.
        """,
       
    ]
    result = [stats_agent.invoke(prompt) for prompt in prompts]
    
    return {# returning output of result to the State Object
        "messages": [
            HumanMessage(content=result[-1]['output'], name="analyst_agent"),
        ]
    }

def file_checker_agent_node(file_path: str) -> str:
    stats_agent_files = ["data_dump/stats/stats.csv", "data_dump/stats/stats_monthly.csv", "data_dump/stats/stats_monthly_avg.csv", "data_dump/stats/stats_thresholds.csv"]
    analyst_agent_files = ["data_dump/analysis/anomaly_report.csv", "data_dump/analysis/anomaly_report_detailed.csv"]
    for file in stats_agent_files:
        if not os.path.exists(file):
            return {
                "messages": [
                    HumanMessage(content="trigger stats_agent", name="file_checker_agent"),
                ]
                }
        else:
            continue
    for file in analyst_agent_files:
        if not os.path.exists(file):
            return {
                "messages": [
                    HumanMessage(content="trigger analyst_agent", name="file_checker_agent"),
                ]
                }
    if not os.path.exists("data_dump/analysis/anomaly_report_story.txt"):
        return {
            "messages": [
                HumanMessage(content="trigger communicator_agent", name="file_checker_agent"),
            ]
            }
    return {# returning output of the file completions to State Object
        "messages": [
            HumanMessage(content="END", name="file_checker_agent"),
        ]
        }

def communicator_agent_node(state: AgentState):#Defining Communication Agent with prompt engineering
    prompt = f"""
        You are a communicator agent, you will be given data in a CSV format.
        The data will contain anomalies that need to be communicated to stakeholders and should be assumed to be invalid.
        Tell a story about the data you have received.
        Your audience is a group of business stakeholders looking for actionable items from anomalous data.
        Here is your dataset: data_dump/analysis/anomaly_report.csv
        Here is a more detailed dataset: data_dump/analysis/anomaly_report_detailed.csv
        Write your message to the stakeholders in a way that is easy to understand and actionable as a text file to data_dump/analysis/anomaly_report_story.txt
        """
    response = stats_agent.invoke(prompt)
    return {
        "messages": [
            HumanMessage(content=response['output'], name="communicator_agent"),
        ]
    }

file_tool = Tool(# Assiging custom function as tool
    name="file_checker_agent",
    func=file_checker_agent_node,
    description="Checks if a file exists."
)
tools = [file_tool]

workflow = StateGraph(MessagesState)#Enabling Stateful transitions
#Registering nodes in Graph
workflow.add_node("stats_agent", stats_agent_node)
workflow.add_node("analyst_agent", analyst_agent_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("file_checker_agent", file_checker_agent_node)
workflow.add_node("communicator_agent", communicator_agent_node)

#Constructing a LangGraph state machine where multiple agent nodes eventually lead to the "supervisor" node
for member in members:
    workflow.add_edge(member, "supervisor")
workflow.add_conditional_edges("supervisor", lambda state: state["next"])
workflow.add_edge(START, "supervisor")

#compiling LangGraph state machine into an executable workflow application.
app = workflow.compile()

#Interactively process the workflow and stream intermediate steps
for s in app.stream(
    {"messages": [("user", """
        Are there any anomalies in the dataset data_dump/data_extract.csv?
        """)]},
    {"recursion_limit": 100},
):
    print(s)
    print("---")
