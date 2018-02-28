FROM jgc128/uml_nlp_class

RUN pip3 install nltk
RUN python3 -m nltk.downloader punkt

ADD . /usr/src/app

CMD ["python3", "train.py"]
