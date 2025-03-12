# Schema Guided Text-to-Table Generation

This project focuses on converting unstructured textual data into structured tables using state-of-the-art models like GPT, Gemini, and Llama. Our approach, **Map&Make (M&M)**, is a structured summarization framework that **dynamically infers table schema** rather than relying on predefined templates. It follows a **three-step process** that extracts key information, structures it into a schema, and fills the tables accurately.

---

## Table of Contents

- [Approach](#approach)
- [Datasets](#datasets)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Evaluation Methods](#evaluation-methods)
- [Comparison Strategies](#comparison-strategies)
- [Insights](#insights)
- [Contributing](#contributing)
- [License](#license)

---

## Approach

The **Map&Make** framework follows a **three-step methodology** to transform text into tables:

1. **Propositional Atomization**  
   - Extracts atomic, self-contained facts from unstructured text.
   - Ensures key properties: grammatical correctness, atomicity, and contextual independence.
   - Eliminates ambiguities, redundant information, and hallucinations.

2. **Schema Extraction**  
   - Identifies and infers the **structure of tables dynamically**.
   - Iteratively maps entities to row headers and attributes to column headers.
   - Adapts schema to diverse datasets without relying on predefined formats.

3. **Table Generation**  
   - Populates extracted schemas using relevant atomic statements.
   - Handles numerical aggregation, categorical attributes, and multi-view information.
   - Ensures consistency and correctness by validating row-column relationships.

This methodology enhances adaptability across different domains and eliminates reliance on predefined schemas.

---

## Datasets

We evaluate our approach on two challenging datasets:

1. **Rotowire**  
   - Contains **NBA post-game summaries (2014–2017)**.
   - Requires extracting performance statistics to generate **Player and Team Tables**.
   - Challenges include **long text, multi-table schema, and sparse data extraction**.

2. **Livesum**  
   - Consists of **live football commentary** that requires **event aggregation**.
   - Tables summarize team performance, tracking goals, assists, and other actions.
   - Columns are labeled **Easy, Medium, or Hard** based on inference difficulty.
   - Challenges include **identifying and aggregating scattered event information**.

Each dataset presents unique challenges in schema induction and table construction.

---

## Setup

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/McHarold404/TableGEN.git
    cd TableGEN
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables:**
    - Create a `.env` file in the root directory.
    - Add the necessary environment variables (API keys for LLMs (Gemini, GPT, Llama)) required for the project.

---

## Project Structure

The repository is organized into the following directories:

- **`batch_scripts/`**  
  Contains scripts for different **table generation methods** (multi-threaded inference)using models like **GPT, Gemini, and Llama**. Includes **schema extraction, atomic statement generation, and one-step/two-step table generation**.

- **`eval/`**  
  Includes evaluation scripts and Jupyter notebooks for **benchmarking** the generated tables using metrics like **Exact Match (EM), CHRF, BERTScore, and TabEval**.

- **`metrics/`**  
  Stores scripts used to compute different **quantitative evaluation metrics** for generated tables.

- **`model_inference/`**  
  Contains helper functions and scripts to **run inference** using LLM models.

- **`model_outputs/`**  
  Stores the **outputs of generated tables** for different models and configurations.

- **`outputs/`**  
  Contains logs and debugging information, including the **process logs**.

- **`post_processing/`**  
  Scripts for **refining and cleaning** generated tables after inference.

- **`prompts/`**  
  Stores prompt templates used for **LLM-based table generation**, structured as zero-shot, few-shot, or CoT.

- **`utils/`**  
  Includes utility functions used across multiple scripts for **data loading, text processing, and visualization**.

- **`Jupyter Notebooks`**  
  Jupyter Notebooks to visualise sample-wise generated tables, qualitative analysis.


---

## Evaluation Methods

We employ **both reference-based and referenceless evaluation metrics** to assess the quality of generated tables.

1. **Rotowire Evaluation**
   - **Exact Match (EM)**: Measures the number of correctly extracted table values.
   - **CHRF (Character n-gram F-score)**: Evaluates similarity between generated and reference tables.
   - **BERTScore**: Computes semantic similarity between generated and reference tables.
   - **TabEval**: A specialized metric that evaluates structural correctness.
   - **Auto-QA**: Uses LLMs to answer questions about the generated tables, testing their completeness.

2. **Livesum Evaluation**
   - **Root Mean Squared Error (RMSE)**: Measures numerical aggregation errors.
   - **Error Rate (ER %)**: Percentage of incorrect cell values in the table.
   - **TabEval & Auto-QA**: Used to assess structured correctness and coverage.

These metrics ensure comprehensive evaluation across **accuracy, completeness, and fidelity**.

---

## Comparison Strategies

To benchmark the **Map&Make (M&M) framework**, we compare it against the following strategies:

1. **Text-Tuple-Table (T3)**
2. **Few-Shot Chain-of-Thought (CoT) Prompting**
3. **Divide & Generate (Planned Addition)**
4. **Supervised Fine-Tuning (Planned Addition)**

Each of these methods provides **a different perspective on schema extraction, table fidelity, and information accuracy**.

---

## Insights

The **Map&Make (M&M) framework** significantly outperformed existing methods in **text-to-table generation** across both **Rotowire and Livesum datasets**. Key findings include:

1. **Better Information Coverage**  
2. **Reduction in Hallucination and Missing Information**  
3. **Improved Event Aggregation in Livesum**  
4. **Generalization Across Large Texts**  
5. **Adaptability Across LLMs**  

These insights validate **Map&Make as a robust, schema-agnostic table generation framework**.

---

## License

This project is licensed under the [MIT License](LICENSE).


