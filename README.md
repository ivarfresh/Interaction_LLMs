# LLM Agents in Interaction: Measuring Personality Consistency and Linguistic Alignment in Interacting Populations of Large Language Models.

Please make sure to:

**1. set your working directory to: “coding/scripts/py”**

**2. Enter your own OpenAI API key in following the data generation files:**

- “Control_groups.py”, “ANACREA_Experimental_groups.py”,“CREAANA_Experimental_groups.py”

**3. Intall Python version and packages**

Python version:

- Python 3.11.4

Required Libraries:

langchain                 0.0.268
numpy                     1.25.2
openai                    0.27.8 
matplotlib-base           3.7.1             
matplotlib-inline         0.1.6
pandas                    2.0.3           
pandas-stubs              1.5.3.230203    
mpmath                    1.3.0
scipy                     1.11.1
seaborn                   0.12.2
sklearn				            1.3.2           
pillow                    9.4.0
requests                  2.31.0
regex                     2023.8.8

**4. Data**

- Our data can be found in “output/Data”

- In order to generate your own data and perform our analyses, continue below

**5. Data generation:**
Main files:
- py/Control_groups.py
- py/CREAANA_Experimental_groups.py
- py/ANACREA_Experimental_groups.py

**5.1 Make sure that the folders: “output/Control”, “output/ANACREA”, “output/CREAANA” exist.**


**5.2 Errors: **

- When runnning the files in 4. you might get a ”ValueError” or “IndexError”. We tried to prevent this by adding error handling, however this is not complete error proof because the GPT model temperature is set to 0.7 (this is needed for variable responses). 

- If this happens, just remove the last rows from the csv files (if something has been saved already) and run it again.


**6. Data Processing: **

- Combining the data files into one file: “BFI_Story_data.csv”.

**6.1 For the Experimental groups**
- Run: merge_subject_groups.py
- Run: combine_csv_files.py  (this will also generate the “BFI_Story_data.csv” file for the control group.)

- Remove all csv files except for “BFI_Story_data.csv”

**6.2 For the Control group**
- Remove all csv files except for “BFI_Story_data.csv”


**6.3 Pre-processing for Point Biserial correlation:**
- Run: PB_encodings.py

**7. To perform the linguistic alignment test:**
- Run: LIWC.py

- note: You must run this before you can perform the correlations

**8. To run the statistical analyses, run the files in ”plot_stats”**


