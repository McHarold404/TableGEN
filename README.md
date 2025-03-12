# Map & Make: Schema Guided Text to-Table Generation

This project focuses on converting unstructured textual data into structured tables using state-of-the-art models like GPT, Gemini, and Llama. Our approach, **Map&Make (M&M)**, is a structured summarization framework that **dynamically infers table schema** rather than relying on predefined templates. It follows a **three-step process** that extracts key information, structures it into a schema, and fills the tables accurately.

---

## Table of Contents

- [Approach](#approach)
- [Datasets](#datasets)
- [Setup](#setup)
- [Usage](#usage)
  - [Generating Atomic Statements and Headers](#generating-atomic-statements-and-headers)
  - [Generating Tables](#generating-tables)
  - [Evaluating Results](#evaluating-results)
- [Evaluation Methods](#evaluation-methods)
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
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables:**
    - Create a `.env` file in the root directory.
    - Add the necessary environment variables required for the project.

---

## Usage

### Generating Atomic Statements and Headers

To generate atomic statements and headers using the Gemini model:

```bash
python batch_scripts/Gemini_Livesum_generate_schema_and_atomic.py
```

### Generating Tables

To generate tables from atomic statements:

```bash
python batch_scripts/GPT_Livesum_TabGen_1step.py --model_dir <model_dir> --model_name <model_name>
```

For a two-step generation process:

```bash
python batch_scripts/GPT_Livesum_TabGen_2step.py --model_dir <model_dir> --model_name <model_name>
```

### Evaluating Results

Run the following notebooks for evaluation:

- `eval/livesum_eval.ipynb`
- `eval/livesum.ipynb`

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

## Contributing

Contributions are welcome! To contribute:

- Open an issue or submit a pull request.
- Follow the project’s guidelines and adhere to the code style.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*Note: Replace placeholders `<repository_url>`, `<repository_name>`, and `<model_dir>` with your actual details before using.*
```
