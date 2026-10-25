# DataSense AI — Dataset Explorer & Insight Generator

A student mini-project built with Python, Streamlit, Pandas, NumPy, and Matplotlib.

## Features
- CSV and Excel upload
- Dataset preview and column profiling
- Missing-value and duplicate detection
- Potential numeric outlier checks using the IQR method
- Automatically generated charts
- Rule-based insight summaries
- Local question-answering for common dataset questions
- Downloadable Markdown report and column profile CSV

## Run on Windows
1. Install Python 3.10+ or use an Anaconda environment.
2. Open this project folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Anaconda Prompt or run:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

5. Install requirements:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Launch the app:

   ```powershell
   python -m streamlit run app.py
   ```

7. Streamlit will display a local URL, usually `http://localhost:8501`. Open it in your browser.

## Important technical note
This version is a transparent local analysis agent: it routes supported question types to Pandas-based calculations and generates insight statements from computed statistics. It does not require an API key and does not send uploaded data to an external AI provider. It does not yet support unrestricted conversational reasoning.

## Suggested demo dataset
Use the included `sample_student_performance.csv`, or upload a relevant CSV/Excel dataset. Explain that the example file is synthetic demonstration data.

## Suggested demonstration sequence
1. Open the app and introduce the agent.
2. Upload the sample dataset.
3. Show overview metrics and preview.
4. Open Data Health and demonstrate missing/duplicate/outlier checks.
5. Show charts under Explore.
6. Ask supported questions in Ask the Agent.
7. Download the report.
