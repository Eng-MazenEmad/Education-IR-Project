# Information Retrieval (IR) Project 2026 | Level 3

A Python & PySpark-based Information Retrieval system implemented over a 10-document corpus (`d1.txt` – `d10.txt`). The project covers text tokenization, positional indexing, term weighting (`TF`, `DF`, `IDF`), and vector normalization (`TF-IDF` with L2 Normalization).

---

## Project Structure

```text
IR_Project/
├── Datasets/                  # Input text documents (d1.txt to d10.txt)
├── Outputs/                   # Generated index and weighting tables
│   ├── positional_index.txt   # Output of Person 1 (Term, Document, Positions)
│   ├── tf_df_idf.txt          # Output of Person 2 (Term, Document, TF, DF, IDF)
│   └── tfidf_normalized.txt   # Output of Person 3 (Term, Document, TF, IDF, TF_IDF, Normalized)
├── Src/                       # Source code and Jupyter Notebooks
│   └── Education.ipynb        # Main implementation notebook
└── README.md                  # Project documentation
```

---

## Team Members & Responsibilities

| Role | Team Member | Task / Module | Output File |
| :--- | :--- | :--- | :--- |
| **Person 1** | Ahmed Ali Mohsen | **Positional Index**: Lowercasing, regex tokenization (`\b\w+\b`) without stop-word removal, and 1-based position tracking per document. | `Outputs/positional_index.txt` |
| **Person 2** | *Member 2 Name* | **TF, DF & IDF**: Calculating Term Frequency per document, Document Frequency across $N=10$ documents, and $\text{IDF}(t) = \log_{10}(N / \text{DF}(t))$. | `Outputs/tf_df_idf.txt` |
| **Person 3** | *Member 3 Name* | **TF-IDF & Normalization**: Computing $\text{TF-IDF} = \text{TF} \times \text{IDF}$ and applying L2 vector normalization per document. | `Outputs/tfidf_normalized.txt` |

*(Note: Replace placeholder names for Person 2 and Person 3 with your teammates' names.)*

---

## Environment & Setup (Windows)

To ensure consistent execution across all team machines without `JAVA_GATEWAY_EXITED` errors, use the following verified environment:

* **OS:** Windows
* **Python:** `3.13`
* **Java:** Java 8 (`jre-1.8`)

### Installation Command
Run this command in the first cell of `Src/Education.ipynb`, then **Restart Kernel**:

```python
%pip install -q --force-reinstall ipykernel pandas pyspark==3.5.3 py4j==0.10.9.7
```

---

## Key Formulas Used

* **Inverse Document Frequency (IDF):**
  $$\text{IDF}(t) = \log_{10}\left(\frac{N}{\text{DF}(t)}\right)$$

* **TF-IDF Weight:**
  $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

* **L2 Normalization:**
  $$\text{Norm}(t, d) = \frac{\text{TF-IDF}(t, d)}{\sqrt{\sum_{k \in d} \text{TF-IDF}(k, d)^2}}$$
