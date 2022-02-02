# Job-Recommender

- Clone the repo
`git clone <SSH>`

- Change to repo's directory
`cd Job-Recommender`

## Set up Environment for the project

- Requires python 3.7 or more.
- Install virtualenv using
`pip install virtualenv`
- Set up python virtual environment for the project, using `virtualenv venv`
- Activate the virtualenv.
- If you are using windows then use the command `./venv/scripts/activate`.
- If you are using linux systems then use command `source venv/bin/activate`
- Download the required dependencies used in the project using, `pip install -r requirements.txt`

Congratulations! You have successfully set up the environment!

## Project Hierarchy
- The  `Datasets/` directory contains all the datasets downloaded for the project. It mainly consists of two job datasets one for IT and other for Non-IT.
- Data preprocessing and natural language processing is done in `code/data_cleaning.py`.
- Saved cleaned data to a new directory `Cleaned_Datasets/` to save time and reduce redundancy during runtime.
- Job recommender is built in `Code/model.py`

## Data cleaning
nltk library used. The following pipeline has been executed.
- remove all non alphabets regex = [^a-zA-Z], 
- remove whitespaces
- convert case to lowercase 
- tokenize words
- remove stopwords
- stemming
The result saved to `Cleaned_Dataset/` directory.

## Model
- First vectorize the text using tfidf vectorizer.
- Then use cosine similarity to find the similarity of text and the one with highest score is the recommended job.

## How to Use:
- First copy the contents of your resume or a resume whose job you want to predict and paste the copied contents in `predictor_files/input.txt`. You can also create your own files, say `random.txt` in the `predictor_files` directory and paste it in there as well.
- Next Open the CLI in repo's home directory.
- cd to `Code/` directory. `cd Code`
- Run the following command to preprocess the raw data. `python data_cleaning.py`. This will clean the datasets and save them in `Cleaned_Datasets/` directory.
- Run the following command if you want to get top 3 recommendations for the resume you input, `python model.py`.

## Various ways of using model.py for job recommendations:
- You can use the `-f <filename>` or `--filename <filename>` options to specify the name of the file in the `predictor_files/` directory that you copied your resume contents to.
- The model points to the `predictor_files/` directory by default for accessing your resume contents so if you are creating new file then create and copy in that directory only.
- Example command - `python -f example1.txt`, `python --filename example1.txt`, note that example1.txt must be in `predictor_files` directory.
- Also, you may not be from IT industry only, you may be a doctor, architect, etc, the model takes into consideration by default that you are from IT industry but you can specify that you are from NON-IT field by using the `-o NON-IT` or `--option NON-IT` flags.
- Example command - to get top 3 recommendations for a NON-IT resume whose contents are pasted in `predictor_files/example2.txt` we use the command `python model.py -o NON-IT -f example2.txt`
- Remember that for theses commands to work your current directory must be `Job-Recommender/Code/`, if its not the same, then cd to it and then use.

## Series of commands
- `cd Code/`
- `python data_cleaning.py`
- Copy resume contents to `predictor_files/input.txt`
- Run `python model.py`

## Example Output:
![image](https://user-images.githubusercontent.com/56588114/145573528-5ad0c6cb-b98f-4be0-a14f-b510fda463fd.png)

