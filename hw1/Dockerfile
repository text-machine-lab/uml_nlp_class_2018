FROM jgc128/uml_nlp_class

RUN apt-get update && apt-get install -y --no-install-recommends  curl unzip  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install fastText
RUN mkdir -p /usr/src/ && \
    git clone https://github.com/facebookresearch/fastText.git /usr/src/fastText && \
    cd /usr/src/fastText && \
    make && \
    cp /usr/src/fastText/fasttext /usr/local/bin && \
    rm -rf /usr/src/fastText

# Download the text8 data
RUN curl -SL http://mattmahoney.net/dc/text8.zip | gunzip > /usr/src/app/text8.txt


# Install pyfasttext
RUN pip3 install cython && pip3 install pyfasttext


ADD . /usr/src/app

CMD ["python3", "hw1.py"]
