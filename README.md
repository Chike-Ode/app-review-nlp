# App Review Analysis

The purpose of this project is to explore HuggingFace's app_reviews dataset. Click [here](https://huggingface.co/datasets/app_reviews) to explore the source data. 

## Getting Started

Setting up your project in a docker container is the easiest way to ensure no package versioning conflicts with your environment. For this project, I used the **jupyter/tensorflow-notebook** docker image which can be found [here](https://hub.docker.com/r/jupyter/tensorflow-notebook)

```bash
1. docker pull jupyter/tensorflow-notebook
2. docker volume create --name *project-name*
3. docker run -it -d -v *project-name*:/home/*project-name* -p 8888:8888 jupyter/tensorflow-notebook
4. docker ps (find the container name tied to the image that was just created)
5. docker start *container-name* (container name from step 4)
6. docker exec -it *container-name* /bin/bash
7. **OPTIONAL** Create anaconda environment
8. git clone https://github.com/Chike-Ode/app-review-nlp.git
9. pip install -r requirements.txt
10. naviguate the notebooks directory for detailed analysis
```

## Project Structure

```bash
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── output         <- Contains scored reviews for each model.
│   └── raw            <- The original data dump split into training, testing and validation.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── images             <- EDA plots and DL model performance loss function
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   └── utils.py       <- Scripts to download or generate data
│   
│
│── evaluation         <- Evaluate model performance
│   ├── graphs         <- Series of graphs for the performance of the models.
│   │
│   └── performance    <- CSV file tracking performance of each model.
```

