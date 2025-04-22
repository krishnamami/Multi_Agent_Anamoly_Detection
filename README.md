                                                                                 
#  Multi-Agent AI Framework for Outlier Detection

## Objective:
A next-generation anomaly detection framework powered by a **Multi-Agent AI architecture**. This system uses intelligent agents that work in parallel to monitor, classify, and escalate anomalies across business metrics enabling near real-time observability and intelligent incident triage.


This project introduces a **multi-agent orchestration model** to::
 * Reduce manual monitoring and investigation
 * Detect issues early with high accuracy
 * Proactively notify stakeholders to prevent revenue and productivity loss

### Outlier Detection Logic :

* Threshold Calculation:

   * For each segment and metric, compute the Monthly Daily Average over the last 6 completed months
     
   * Define thresholds as:
     
       -->Upper Threshold = 200% of highest historical daily average maximum
     
       -->Lower Threshold = 50% of lowest historical daily average minimum
  
 * Detection Window:

    * Apply the thresholds to the most recent 15-day window
    * Detect anomalies across all segments and metrics

## Architecture


   ![image](https://github.com/user-attachments/assets/e9d84ac7-ae21-47b1-a8b9-6f96ecb2531f)


   
### Agent Roles & Responsibilities

* Supervisor Agent :
   * Co-ordinates and monitors the execution of all agents.
* Stats Agent :
   * Cleans data, calculates monthly daily averages, and defines thresholds per segment.
* Analyst Agent :
   * Processes recent data to compute daily averages and compares them against thresholds.
* File Checker Agent:
   * Verifies the completion of Stats and Analyst tasks and reports back to the Supervisor Agent.

### Tech Stack:
- **Languages**: Python  
- **Frameworks**: LangGraph, LangChain 
- **Infrastructure**: Airflow

 
### Outcomes

* Hands-on experience with LangGraph and LangChain and its architecture, including agent state management.
* Effective for non-critical production applications.
* Inspired the creation of an AI-powered incident management system for Data Engineers:
   * Achieved 30% improvement in engineering productivity.
   * Reduced manual debugging and resolution time for production failures

### Next Steps 

* Integrate Agentic RAG (Retrieval-Augmented Generation) for contextual decisioning.
* Explore managed service deployment with observability and guardrails.

##  How to Run (Local Simulation)

git clone https://github.com/krishnamami/Multi_Agent_Anamoly_Detection.git cd Multi_Agent_Anamoly_Detection python agents.py

## Related Projects
[Fine_Tuning_LLM](https://github.com/krishnamami/Fine_Tuning_LLM)

[Markov_Chain_Attribution](https://github.com/krishnamami/Markov_Chain_Attribution)

[Distributed ML SageMaker Pipeline](https://github.com/krishnamami/Distributed_ML_Sagemaker_Pipelines)

## Author
Krishna Goud

Head of Data Engineering & MLOps | Rocket LA   [LinkedIn](https://www.linkedin.com/in/krishnagoud)

Delivering $4B+ business impact via AI-first, scalable, real-time data systems



  
