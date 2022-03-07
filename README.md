# App Review Analysis

The purpose of this project is to explore HuggingFace's app_reviews dataset. Click [here](https://huggingface.co/datasets/app_reviews) to explore the source data. 

## Getting Started

Setting up your project in a docker container is the easiest way to ensure no package versioning conflicts with your environment. For this project, I used the **jupyter/tensorflow-notebook** docker image which can be found [here](https://hub.docker.com/r/jupyter/tensorflow-notebook)

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

