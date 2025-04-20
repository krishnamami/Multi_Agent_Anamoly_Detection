                                                                                 
#  Multi-Agent AI Framework for Outlier Detection

## Objective:
Implement a Multi-Agent AI framework to automate the detection of outliers and data deviations using deterministic methods. This system reduces manual intervention, enhances detection accuracy, and delivers faster, actionable insights. Detected anomalies are validated to prevent false positives and are proactively communicated to stakeholders for timely decision-making.

### Outlier Detection Logic :

* Threshold Calculation:

  For each segment and metric, compute the Monthly Daily Average over the last 6 completed months
  
  Define thresholds as:
  
   -->Upper Threshold = 200% of historical daily average maximum
  
   -->Lower Threshold = 50% of historical daily average minimum
  
 * Detection Window:

   Apply thresholds to identify anomalies in the most recent 15-day window, across all segments and metrics.

## Architecture


   ![image](https://github.com/user-attachments/assets/e9d84ac7-ae21-47b1-a8b9-6f96ecb2531f)


   
### Agent Roles & Responsibilities

* Supervisor Agent :
   * Orchestrates overall execution, monitors agent performance, and execution.
* Stats Agent :
   * Cleans data, calculates monthly daily averages, and assigns segment-level threshold values.
* Analyst Agent :
   * Processes recent data to compute daily averages and compares them against thresholds.
* File Checker Agent:
   * Ensures Stats and Analyst Agents complete their tasks and reports status to the Supervisor.
 
### Learnings

* Gained hands-on experience with LangGraph and LangChain, including state management across agents.
* Proven effective for non-critical production applications with potential for scale.
* Inspired AI-driven incident management system tailored for Data Engineers, enabling automated resolution of production failures and achieving a 30% boost in productivity.

### Next Steps 

* Integrate Agentic RAG (Retrieval-Augmented Generation) for knowledge-driven decisions and feedback loops.
* Explore Managed Services and accelerate deployment with built-in guardrails for reliability and observability.

  
