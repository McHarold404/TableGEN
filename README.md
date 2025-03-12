# Livesum Table Generation Project

This project focuses on converting unstructured textual data into structured tables using state-of-the-art models like GPT, Gemini, and Llama. Our approach, **Map&Make (M&M)**, is a structured summarization framework that **dynamically infers table schema** rather than relying on predefined templates. It follows a **three-step process** that extracts key information, structures it into a schema, and fills the tables accurately.

---

## Table of Contents

- [Approach](#approach)
- [Datasets](#datasets)
- [Setup](#setup)
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
   - Converts input text into structured tuples (**subject-object-verb** or **subject-attribute-value**).
   - Generates tables based on extracted tuples.
   - Originally designed for predefined schemas, later adapted for schema-agnostic settings.

2. **Few-Shot Chain-of-Thought (CoT) Prompting**
   - Uses few-shot learning to generate tables.
   - Employs step-by-step reasoning to ensure logical coherence.
   - Improves over Zero-Shot prompting but struggles with schema variability.

3. **Divide & Generate (Planned Addition)**
   - Splits input text into smaller, more manageable chunks.
   - Independently processes each segment before integrating the results.
   - Expected to improve schema induction and minimize hallucinations.

4. **Supervised Fine-Tuning (Planned Addition)**
   - Involves fine-tuning LLMs on labeled table-generation datasets.
   - Aims to enhance table fidelity by reinforcing specific formatting rules.
   - Requires a large volume of training data and extensive model tuning.

These approaches provide a **comprehensive benchmark** for evaluating M&M’s effectiveness in **text-to-table conversion**.

---

## Insights

The **Map&Make (M&M) framework** significantly outperformed existing methods in **text-to-table generation** across both **Rotowire and Livesum datasets**. Key findings include:

1. **Better Information Coverage**  
   - M&M outperformed **Zero-Shot and One-Shot CoT baselines** by **up to 32%** in accuracy (CHRF) and **42% in structured correctness (TabEval)**.
   - Compared to prior methods like **Text-Tuple-Table (T3)**, M&M achieved **higher row and column coverage**.

2. **Reduction in Hallucination and Missing Information**  
   - Hallucinated columns were reduced by **up to 60%**, ensuring a **higher fidelity** in table generation.
   - M&M’s schema induction strategy improved **correct entity mappings**, reducing missing values in Rotowire tables.

3. **Improved Event Aggregation in Livesum**  
   - **Error rates (ER%) decreased by 35%** in a zero-shot setting and **55% in one-shot settings**.
   - **RMSE (Root Mean Squared Error) improved by up to 57%**, ensuring better numerical consistency in **live commentary event tracking**.

4. **Generalization Across Large Texts**  
   - M&M maintained performance stability **even with larger input text**, unlike **standard CoT prompting**, which showed degradation.
   - Schema extraction remained consistent across **multi-table formats** in Rotowire and **single evolving table structures** in Livesum.

5. **Adaptability Across LLMs**  
   - Performance gains were observed across **GPT-4o, Gemini 2.0, and Llama-3.3 70B**.
   - **Gemini 2.0 exhibited a slight edge in structured correctness**, while **GPT-4o excelled in information coverage**.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

This version **clarifies the benchmarking strategies** used to compare M&M with **alternative approaches**, while maintaining a clean, structured, and easy-to-read format. 🚀 Let me know if you'd like any refinements!
