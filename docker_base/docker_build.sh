#!/usr/bin/env bash

docker build -t uml_nlp_class .
docker tag uml_nlp_class jgc128/uml_nlp_class
