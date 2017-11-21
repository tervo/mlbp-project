FROM tensorflow/tensorflow

RUN mkdir /a
WORKDIR /a

RUN apt-get update
RUN apt-get install -y python-tk wget graphviz
RUN pip install matplotlib keras pydot graphviz keras-vis
RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh && bash Miniconda2-latest-Linux-x86_64.sh -b
RUN /root/miniconda2/bin/conda install theano pygpu mkl-service
ENV PYTHONPATH=/root/miniconda2/lib/python2.7/site-packages

# docker run --rm -v /Users/tervo/Dropbox/phd/kurssit/CS-E3210-Machine_Learning_Basic_Principles/project:/a tervo/mlbp-project project.py
# docker run -e THEANO_FLAGS='exception_verbosity=high' --rm -v /Users/tervo/Dropbox/phd/kurssit/CS-E3210-Machine_Learning_Basic_Principles/project:/a tervo/mlbp-project python -u nn.py