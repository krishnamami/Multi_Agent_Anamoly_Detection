                                                                                 
#  Multi-Agent AI Framework for Outlier Detection

## Objective:
This project demonstrates a Multi-Agent AI Framework designed to automate the detection of data anomalies and deviations using deterministic methods. 

The system is built to:
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
   * Verifies the completion of Stats and Analyst tasks and reports back to the Superviso.
 
### Learnings

* Hands-on experience with LangGraph and LangChain, including agent state management.
* Effective for non-critical production applications.
* Inspired the creation of an AI-powered incident management system for Data Engineers:
   * Achieved 30% improvement in engineering productivity.
   * Reduced manual debugging and resolution time for production failures

### Next Steps 

* Integrate Agentic RAG (Retrieval-Augmented Generation) for contextual decisioning.
* Explore managed service deployment with observability and guardrails.


  
