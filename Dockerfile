FROM continuumio/miniconda3:latest
COPY . /app
WORKDIR /app
RUN conda install keras
RUN conda install scikit-learn
RUN pip install -r requirements.txt
EXPOSE 5802
CMD python ./flask_api.py
