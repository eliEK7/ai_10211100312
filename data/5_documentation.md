# Documentation (LLM)
A guide containing detailed descriptions of the datasets and models used, and a breakdown of the methodology behind the LLM-based question and answer functionality. Additionally, this section outlines the architecture of the LLM task, explaining the workflow of data processing and model interaction. It also provides an evaluation of the results, comparing the output of the developed model with that of ChatGPT to assess its performance. 

## Author Information
    Evans Eli Kumah - 10211100312
    evans.kumah@acity.edu.gh

## Table of Contents
- [Methodology (LLM)](#methodology-llm)
  - [Model Selection](#model-selection)
  - [Document Preparation](#document-preparation)
  - [Text Chunking](#text-chunking)
  - [User Query Handling](#user-query-handling)
  - [Response Generation](#response-generation)
  - [Response Time Measurement](#response-time-measurement)
  - [Confidence Score](#confidence-score)
  - [Display of Results](#display-of-results)
  - [Interaction with Users](#interaction-with-users)
  - [Key Components](#key-components-of-the-system)
- [System Architecture (LLM)](#system-architecture-llm)
  - [Document Extraction & Chunking](#1-document-extraction--chunking-backend)
  - [Frontend - Ask a Question](#2-frontend---ask-a-question-user-interface)
  - [Model Query Processing](#3-model-query-processing-backend)
  - [Confidence Calculation](#4-confidence-calculation-backend)
  - [Chat History & Session State](#5-chat-history--session-state-backend)
  - [Frontend - Display Results](#6-frontend---display-results-streamlit-app)
  - [Technologies Used](#technologies-used)
  - [Flow Overview](#flow-overview)
- [Evaluation, Analysis and Comparison with ChatGPT](#evaluation-analysis-and-compararison-with-chatgpt)
  - [AI Model Comparison](#-ai-model-comparison-2025-ghana-budget-analysis)
  - [Overall Summary](#overall-summary)
  - [Responses](#responses)

## Methodology (LLM)
The methodology for our Large Language Model (LLM) solution relies on leveraging Mistral-7B-Instruct-v0.3, a pre-trained open-source model, to perform natural language processing tasks such as Question & Answer (Q&A) from custom data. 

1. Model Selection: The Mistral-7B-Instruct-v0.3 model, a pre-trained open-source large language model (LLM), is used to generate responses to user queries. This model is optimized for instruction-based tasks and has been fine-tuned to understand and respond to complex questions in natural language.

    **Model Description**
    - Name:
        mistralai/mistral-7b-instruct-v0.3

    - Type:
        Instruction-tuned Large Language Model (LLM)

    - Parameters:
        7 Billion

    - Capabilities:
        - Supports instruction-following in question-answering, summarization, and dialogue.
        - Low-latency, open-source alternative to GPT-style models.
        - Fine-tuned for higher factual accuracy and task clarity.

    - API Access:
        - Hosted by NVIDIA via their LLM endpoint.
        - Accessible using OpenAI-compatible SDK via openai.OpenAI(base_url=..., api_key=...)

2. Document Preparation: I utilized the Budget Statement and Economic Policy of the Government of Ghana for the 2025 Financial Year PDF as our custom data source. The PDF document is extracted using the PyMuPDF library (imported as fitz), which allows us to read the content of the PDF file and convert it into text for the model to process.

    **Dataset Description**
    - Source:
        "The Budget Statement and Economic Policy of the Government of Ghana for the 2025 Financial Year"

    - Downloaded from:
        https://mofep.gov.gh/sites/default/files/budget-statements/2025-BudgetStatement-and-Economic-Policy_v4.pdf

    - Description:
        Official budget document published by the Ministry of Finance.

        Contains ~100+ pages of text, tables, and structured policy information.

    - Covers:
        - Fiscal projections
        - Economic plans
        - Sector allocations
        - Development goals

    Language is formal and domain-specific.
    Suitable for question-answering on national budget priorities and policies.

    - Preprocessing Summary:
        - Extracted with fitz for better layout retention.
        - Cleaned and chunked using word-based logic (~3500 tokens per chunk).
        - Stored in memory for fast access during inference.

3. Text Chunking: Since LLMs have a token limit for input, and the document may be too large to process in one go, I split the extracted text into smaller chunks. Each chunk is limited by the model's maximum token capacity, ensuring that no single chunk exceeds the limit. The chunks are generated by iterating through the words in the document and grouping them until the token count reaches the specified limit.
    - Token Limit: The model has a token limit (Mistral-7B-Instruct-v0.3 supports around 3500 tokens). One token corresponds roughly to a word or part of a word.

    - Chunking Strategy:
        - The text is split into words first, then grouped into chunks.
        - Iterate through the words and add them to the current chunk, counting tokens until the chunk exceeds the token limit.
        - Once the limit is reached, the chunk is stored, and a new chunk begins.

4. User Query Handling: When a user submits a question, the system retrieves one of the text chunks to serve as the context for the LLM. The question is then sent to the Mistral-7B-Instruct-v0.3 model, along with the context text, to generate a response.
    - Context Selection: The system selects one of the document chunks as context to provide the LLM. In a more advanced version, the most relevant chunk could be selected using a search or ranking algorithm (e.g., TF-IDF, BM25, or cosine similarity).

5. Response Generation: The query and context are sent to the Mistral API through the NVIDIA endpoint, where the model processes the input and returns a response. The response is streamed, meaning the model sends partial content as it generates the output, which is collected and combined to form the full answer.
    - API Request: The client.chat.completions.create() method is used to send the query and context to Mistral-7B-Instruct-v0.3. The model uses a temperature setting of 0.2 for deterministic responses, reducing randomness.

    - Streaming: The stream=True parameter allows the model to send partial content as it's generated, which we then concatenate to form the full response.

6. Response Time Measurement: The response time is tracked using Python's time library. The start time is recorded before querying the model, and the end time is recorded after receiving the response. The difference between the start and end time gives the response time.
    - Time Tracking: The response time is calculated using Python's time library. The start time is recorded before making the API call, and the end time is recorded after the response is received.

    - Confidence Proxy: A shorter response time is assumed to indicate higher confidence in the model's response. The time difference is used to determine the confidence score.

7. Confidence Score: Since Mistral-7B-Instruct-v0.3 does not provide an explicit confidence score, I use the response time as a proxy for confidence. The assumption is that a faster response indicates higher confidence. The confidence score is calculated as follows:

    - If the response time is less than 5 seconds, a confidence score of 1.0 is assigned (indicating high confidence).

    - If the response time is greater than 5 seconds, the confidence score decreases based on the logarithm of the response time. This reflects the idea that longer response times are less reliable.

8. Display of Results: The model's response, confidence score, and response time are displayed to the user via the Streamlit app. The chat history is maintained so that users can view both their questions and the model's answers in a conversational format. The confidence score and response time are updated and shown after each query.

9. Interaction with Users: The system allows users to ask any question related to the 2025 Budget Statement document. The query is processed, and the model provides answers based on the content of the document. Additionally, the system allows for multiple rounds of questions and answers, with the chat history growing over time.
    - Session State: Streamlit’s st.session_state is used to store the chat history, so users can have ongoing conversations with the model.

    - Chat History: Each interaction (both user and bot messages) is stored and displayed in the chat window. The messages are rendered with HTML styling for a conversational format.

- **Key Components of the System:**
    - Mistral-7B-Instruct-v0.3: The core model for generating responses.
    - PyMuPDF (fitz): Used to extract and process text from the PDF document.
    - Text Chunking: To handle large documents and prevent token overflow.
    - Streamlit: Provides an interactive web interface for users to input queries and view responses.
    - Time Tracking: Measures response time and uses it as a proxy for confidence.
    - Confidence Calculation: Relies on response time to infer the model's confidence in the response.


## System Architecture (LLM)
This system facilitates a Q&A interface using a large language model (LLM) to interact with a PDF document, providing users with responses and calculated confidence scores.

![System Architecture Image](data/architecture.png)

 The architecture is designed with the following key components:

#### 1. Document Extraction & Chunking (Backend)
    Description:
    - Text Extraction: Extracts raw text from the provided PDF document using PyMuPDF
    - Text Chunking: Splits the extracted text into manageable chunks to avoid exceeding model token limits

    Key Components:
    - PDF Text Extraction: Uses PyMuPDF (fitz) to extract text from the PDF document
    - Text Chunking: Creates smaller, manageable text chunks within token limits

#### 2. Frontend - Ask a Question (User Interface)
    Description:
    - Provides user interface via Streamlit app
    - Handles question submission and backend communication

    Key Components:
    - Text Input: Clean interface for entering questions
    - Question Handling: Routes questions to backend processor

#### 3. Model Query Processing (Backend)
    Description:
    - Prepares queries and context for the Mistral-7B Instruct model
    - Handles real-time response processing and timing

    Key Components:
    - API Query: Combines user query with document context
    - Response Handling: Processes streamed model responses
    - Response Time Measurement: Tracks query processing duration

#### 4. Confidence Calculation (Backend)
    Description:
    - Calculates confidence scores based on response times
    - Implements logarithmic scoring system

    Key Components:
    - Response Time Calculation: Measures model response duration
    - Confidence Score: Converts time to confidence metric

#### 5. Chat History & Session State (Backend)
    Description:
    - Maintains conversation history and session context
    - Enables seamless multi-turn interactions

    Key Components:
    - Session State: Tracks ongoing conversation context
    - History Display: Shows previous Q&A exchanges

#### 6. Frontend - Display Results (Streamlit App)
    Description:
    - Renders model responses and metadata
    - Provides interactive chat interface

    Key Components:
    - Response Display: Shows model answers
    - Confidence & Response Time: Displays performance metrics
    - Chat History: Shows scrollable conversation log

#### Technologies Used

**Backend:**
- PyMuPDF (fitz): PDF text extraction
- OpenAI API (NVIDIA endpoint): Model interaction
- Python libraries: Time management, calculations
- Streamlit: Session and state management

**Frontend:**
- Streamlit: UI components and state management

#### Flow Overview

**Backend Flow:**
1. Document Processor:
   - Extracts PDF text
   - Creates text chunks

2. Model Query Processor:
   - Prepares and sends queries
   - Measures response times

3. Confidence Calculator:
   - Calculates confidence scores
   - Processes timing data

4. Chat History:
   - Maintains session state
   - Stores conversation history

**Frontend Flow:**
1. User Input:
   - Question entry interface
   - Input validation

2. Backend Interaction:
   - Query processing
   - Response handling

3. Display Results:
   - Shows model responses
   - Displays metrics and history


##  Evaluation, Analysis and Compararison with ChatGPT

### 📊 AI Model Comparison: 2025 Ghana Budget Analysis
**Find the full responses below**

### Q1: Who Presented the Budget?

| Criteria         | ChatGPT | Budget Bot |
|------------------|---------|------------|
| ✅ Accuracy       | ✅ Correctly identifies Dr. Cassiel Ato Forson and Mahama | ✅ Same info provided |
| ✅ Completeness   | ✅ Includes date, authority, and political context | ✅ Includes Act reference and theme |
| ✅ Clarity        | ✅ Clear and concise | Slightly verbose, but clear |
| ✅ Extra Insights | ❌       | ✅ Mentions public access and website link |

**Verdict**: Both models got it right.  
**Note**: Budget Bot provided extra information so the response may be more insightful but this is subjective because some people will prefer a straight forward answer.

---

### Q2: Key Takeaways from the Budget

| Criteria         | ChatGPT | Budget Bot |
|------------------|---------|------------|
| ✅ Accuracy       | ✅ Accurate across fiscal, macro, and policy initiatives | ✅ Mostly accurate, though some items feel vague or repetitive |
| ✅ Completeness   | ✅ Covers wide breadth: economy, taxes, social policies, reforms | ⚠️ Focuses more on legal reform and policy repetition (same acts repeated multiple times) |
| ✅ Clarity        | ✅ Very clear with structured sections and icons | ⚠️ List format good, but redundant entries (points 10–21) hurt readability |
| ✅ Organization   | ✅ Grouped by theme (macroeconomy, reforms, etc.) | ⚠️ Simple numbered list, but repetitive and ungrouped |
| ✅ Extra Insight  | ✅ Includes macroeconomic targets and outcomes | ⚠️ Lacks specifics like GDP targets, inflation, fiscal deficit, etc. |

**Verdict**: ChatGPT provided a better response.  
**Note**:
- ChatGPT was structured, diverse, clearer. 
- Budget Bot was repetitive, scattered and unstructured.


### Q3: Key Tax Policy Changes

| Criteria         | ChatGPT | Budget Bot |
|------------------|---------|------------|
| ✅ Accuracy       | ✅ Lists tax reforms announced in the actual budget | ⚠️ Partially accurate — includes DST which is not mention in the budget |
| ✅ Completeness   | ✅ Comprehensive: abolished taxes, VAT reforms, levies, admin reforms | ⚠️ Misses abolished taxes and VAT reforms |
| ✅ Clarity        | ✅ Structured with headings and icons | ✅ Clear and well-broken down by bullet points |
| ✅ Organization   | ✅ Highly organized into sections | ✅ Organized by theme (e.g., revenue, expenditure) |
| ✅ Extra Insight  | ✅ Provides policy rationale and stakeholder impact | ✅ Includes broader fiscal strategy and resource allocation context |

**Verdict**: ChatGPT gave a better breakdown of the actual budget policy changes.  
**Note**:
- Budget Bot provided a response on  Digital Services Tax (DST) which is not in the budget


## Overall Summary

| Question | Winner     | Why                                           |
|----------|------------|-----------------------------------------------|
| Q1       | Tie        | Both are accurate; yours added formality      |
| Q2       | ChatGPT    | Better clarity, structure, and variety        |
| Q3       | ChatGPT    | More accurate and specific to real changes    |


### Responses 

#### Budget Bot
    Q1:  The budget was presented by Dr. Cassiel Ato Forson (MP), the Minister for Finance, on behalf of His Excellency John Dramani Mahama, President of the Republic of Ghana, on Tuesday, March 11, 2025. The budget statement and economic policy of the Government of Ghana for the 2025 Financial Year was presented to Parliament in accordance with Section 28 of the Public Financial Management Act, 2016 (Act 921). The budget is available for public access, and electronic copies can be downloaded from the Ministry of Finance's website (www.mofep.gov.gh). The theme of the budget is "Resetting the Economy for the Ghana We Want."

    Q2: The key takeaways from the 2025 Budget Statement and Economic Policy of the Government of Ghana, presented by Dr. Cassiel Ato Forson (MP), Minister for Finance, are as follows:
    1.	The budget theme is "Resetting the Economy for the Ghana We Want."
    2.	The budget aims to address the challenges of unemployment, economic instability, and social justice, as well as to build a Ghana that provides opportunities for all.
    3.	The budget was preceded by a National Economic Dialogue and extensive consultations with various stakeholders, including traders, the business community, and the youth.
    4.	The budget outlines far-reaching measures to manage exchange rate and inflation effectively and efficiently, in close collaboration with the Bank of Ghana.
    5.	The budget includes measures to address the issue of pharmacists who have not received their salaries since June 2023.
    6.	The budget proposes a review of several Acts to support the policies of the government aimed at stabilizing the economy and promoting inclusive growth.
    7.	The budget also includes a commitment to the 24-Hour Economy policy aimed at stimulating economic growth and job creation.
    8.	The budget proposes the establishment of the Ghana Gold Board to provide a legal framework for the accumulation of foreign exchange and gold reserves.
    9.	The budget also includes a commitment to review the Labour Act, the Ghana Investment Promotion Centre Act, and other enabling legislations to support the 24-Hour Economy policy.
    10.	The budget proposes a review of the Petroleum Revenue Management Act, the Ghana Infrastructure Investment Fund Act, the Minerals Income Investment Fund Act, the Energy Sector Levies Act, the Ghana Cocoa Board Act, the Earmark Funds Capping and Realignment Act, the Public Procurement Authority Act, the Ghana Education Trust Fund Act, the Fiscal Responsibility Act, and the Revenue Administration Act.
    11.	The budget also includes a commitment to incorporate these Acts into a comprehensive Public Financial Management Act.
    12.	The budget proposes a review of the Petroleum Revenue Management Act, the Ghana Infrastructure Investment Fund Act, the Minerals Income Investment Fund Act, the Energy Sector Levies Act, the Ghana Cocoa Board Act, the Earmark Funds Capping and Realignment Act, the Public Procurement Authority Act, the Ghana Education Trust Fund Act, the Fiscal Responsibility Act, and the Revenue Administration Act.
    13.	The budget also includes a commitment to incorporate these Acts into a comprehensive Public Financial Management Act.
    14.	The budget proposes a review of the Labour Act, the Ghana Investment Promotion Centre Act, and other enabling legislations to support the 24-Hour Economy policy.
    15.	The budget also includes a commitment to review the Petroleum Revenue Management Act, the Ghana Infrastructure Investment Fund Act, the Minerals Income Investment Fund Act, the Energy Sector Levies Act, the Ghana Cocoa Board Act, the Earmark Funds Capping and Realignment Act, the Public Procurement Authority Act, the Ghana Education Trust Fund Act, the Fiscal Responsibility Act, and the Revenue Administration Act.
    16.	The budget also includes a commitment to incorporate these Acts into a comprehensive Public Financial Management Act.
    17.	The budget proposes a review of the Labour Act, the Ghana Investment Promotion Centre Act, and other enabling legislations to support the 24-Hour Economy policy.
    18.	The budget also includes a commitment to review the Petroleum Revenue Management Act, the Ghana Infrastructure Investment Fund Act, the Minerals Income Investment Fund Act, the Energy Sector Levies Act, the Ghana Cocoa Board Act, the Earmark Funds Capping and Realignment Act, the Public Procurement Authority Act, the Ghana Education Trust Fund Act, the Fiscal Responsibility Act, and the Revenue Administration Act.
    19.	The budget also includes a commitment to incorporate these Acts into a comprehensive Public Financial Management Act.
    20.	The budget proposes a review of the Labour Act, the Ghana Investment Promotion Centre Act, and other enabling legislations to support the 24-Hour Economy policy.
    21.	The budget also includes a commitment to review the Petroleum Revenue Management Act, the Ghana Infrastructure Investment Fund Act, the Minerals Income Investment Fund Act, the Energy Sector Level

    Q3:  Based on the provided context, the key tax policy changes in the 2025 Budget Statement and Economic Policy of the Government of Ghana, as presented by Dr. Cassiel Ato Forson (MP), Minister for Finance, are as follows:
    1.	Fiscal Policy Objectives: The budget aims to achieve a primary balance of 0.5% of GDP, a fiscal deficit of 4.5% of GDP, and a medium-term fiscal consolidation path towards a sustainable debt-to-GDP ratio of 60%.
    2.	Revenue Measures:
    - Introduce a Digital Services Tax (DST) of 2% on the gross transaction value of specified digital services provided by non-resident companies.
    - Increase the National Health Insurance Levy (NHIL) from 2.5% to 3% on the taxable value of goods and services.
    - Implement a comprehensive review of tax exemptions and incentives to ensure they are aligned with the country's development priorities.
    3.	Expenditure Measures:
    - Implement a comprehensive review of public sector wages and salaries to ensure efficiency and fairness.
    - Increase capital expenditure to support infrastructure development, particularly in the areas of energy, transportation, and digital infrastructure.
    4.	Resource Allocation:
    - Increase allocation to the education sector to improve access and quality of education.
    - Increase allocation to the health sector to improve healthcare services and infrastructure.
    - Increase allocation to the agriculture sector to support smallholder farmers and promote agricultural transformation.
    5.	Debt Policy Objectives and Liability Management:
    - Implement a medium-term debt management strategy to ensure debt sustainability and maintain a prudent debt management framework.
    -Prioritize domestic borrowing to reduce reliance on external financing and promote financial stability.
    6.	Key Initiatives and Resource Allocation:
    - Implement the Ghana CARES Obaatanpa Programme to support businesses and households affected by the COVID-19 pandemic.
    - Implement the Ghana Infrastructure Investment Fund (GIIF) to mobilize private sector investment for infrastructure development.
    - Implement the Ghana Petroleum Funds (GPFs) to ensure sustainable management of petroleum revenues for future generations.
    These policy measures are aimed at resetting the economy, promoting inclusive growth, and achieving the vision of the Ghana We Want. The budget also emphasizes the importance of transparency, accountability, and effective public financial management in achieving these objectives.


#### ChatGPT
    Q1: The 2025 Budget Statement and Economic Policy of the Government of Ghana was presented to Parliament by Dr. Cassiel Ato Forson (MP), Minister for Finance, on Tuesday, March 11, 2025, on the authority of His Excellency John Dramani Mahama, President of the Republic of Ghana​

    Q2: Here are the key takeaways from Ghana’s **2025 Budget Statement and Economic Policy** presented by Dr. Cassiel Ato Forson:

    ---

    ### 🎯 **Macroeconomic Targets for 2025**
    - **GDP Growth**: At least **4.0%** (non-oil real GDP at 4.8%)
    - **Inflation**: Targeted to reduce to **11.9%** from 23.8% in 2024
    - **Primary Balance**: A surplus of **1.5% of GDP**
    - **Reserves**: Enough to cover **at least 3 months of imports**

    ---

    ### 💡 **Key Policy Interventions**
    1. **$10 Billion “Big Push” Infrastructure Agenda**:
    - Aimed at massive public infrastructure for job creation and economic transformation.

    2. **24-Hour Economy Initiative**:
    - Encourage round-the-clock operations in sectors to boost productivity and employment.

    3. **Establishment of Ghana Gold Board (GOLDBOD)**:
    - To manage gold exports and build forex reserves.

    4. **Women’s Development Bank**:
    - Provide tailored financial services and low-interest loans to women-led businesses.

    5. **National Apprenticeship Programme**:
    - Free technical/vocational training for youth, backed by startup capital.

    6. **‘Adwumawura’ Programme**:
    - Mentorship and funding support for 10,000 youth-led startups.

    7. **$3 Billion Digital Jobs Initiative**:
    - Includes One Million Coders Programme, FinTech Growth Fund, and Zonal ICT Parks.

    8. **Agriculture for Economic Transformation Agenda (AETA)**:
    - Support food security, reduce food inflation, and create jobs.

    ---

    ### 🧮 **Fiscal Strategy**
    - **Frontloaded fiscal adjustment** of **5.3% of GDP**, the most ambitious in Ghana's history.
    - **Primary expenditure** projected to drop from 19.8% of GDP (2024) to 14.6% (2025).
    - **Deficit reduction** from 7.9% (2024) to 3.1% (2025), aiming for a **0.1% surplus by 2028**.

    ---

    ### 🧱 **Structural and Governance Reforms**
    - Downsizing government ministries from 30 to 23.
    - Reduction of ministers from 123 to 60.
    - Amendment of Fiscal Responsibility Act to include a **debt rule** and establish an **independent fiscal council**.

    ---

    ### 🌍 **Outlook**
    - Positive external sector driven by gold exports, high remittances, and strong commodity prices.
    - Ongoing IMF-supported program expected to stabilize investor confidence and improve FDI flows.

    Let me know if you want the sector-specific allocations or breakdowns too.

    Q3: Here are the **key tax policy changes** proposed in Ghana’s **2025 Budget Statement**:

    ---

    ### ✅ **Abolished Taxes ("Nuisance Taxes")**
    The government plans to eliminate the following taxes in line with its manifesto:
    1. **10% Withholding Tax on lottery winnings** ("Betting Tax")
    2. **1% Electronic Transfer Levy (E-Levy)**
    3. **Emission Levy** on industries and vehicles
    4. **VAT on motor vehicle insurance**
    5. **1.5% Withholding Tax on unprocessed gold** from small-scale miners

    ---

    ### 📉 **VAT Reforms**
    A major overhaul is planned to reduce burdens and streamline the VAT system:
    - **Abolish the COVID-19 Levy (1%)**
    - **Re-couple GETFund and NHIL with VAT**, allowing input VAT claims
    - **Reduce the effective VAT rate** (currently ~22%)
    - **Reverse the VAT Flat Rate Scheme**
    - **Raise VAT registration threshold** to exempt micro/small businesses
    - **Improve compliance through education and tech adoption**

    ---

    ### 💰 **Increased/Modified Levies**
    - **Growth & Sustainability Levy** on mining to rise from **1% to 3%** (reflect gold price increases)
    - **Extend the Special Import Levy and Growth & Sustainability Levy** sunset clauses to **2028**
    - **Review and reintroduce road tolls** via a **technology-driven solution**
    - **Consolidate energy-related levies** into a single levy to address energy sector debt

    ---

    ### 🧾 **Other Tax Administration Reforms**
    - Overhaul of the **entire tax system**
    - Review of **port taxes/fees** to reduce cost of doing business
    - Digitization of the **Modified Taxation System**
    - **Voluntary Disclosure Programme** extended with waivers for penalties
    - Elimination of **VAT exemptions on non-life insurance** (excluding motor)
    - **Tax refund abuse clampdown**: Refund rate reduced from 6% to 4% of revenue

    ---

    Let me know if you want a breakdown by sector (like mining, energy, digital) or how these affect consumers and businesses.
