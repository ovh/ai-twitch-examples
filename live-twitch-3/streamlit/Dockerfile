FROM ubuntu:20.04

# set working directory
WORKDIR /workspace

# install pip
RUN apt-get update && \
	apt-get install -y python3-pip

RUN chown -R 42420:42420 /workspace && \
	addgroup --gid 42420 ovh && \
	useradd --uid 42420 -g ovh --shell /bin/bash -d /workspace ovh

USER ovh

ENV PATH /workspace/.local/bin:$PATH

ADD --chown=ovh:ovh ./requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ADD --chown=ovh:ovh . /workspace

# run server
CMD streamlit run myapp.py --server.headless true