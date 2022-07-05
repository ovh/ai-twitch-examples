FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# expose
EXPOSE 8080

# set working directory
WORKDIR /workspace

# install pip
RUN apt-get update && \
	apt-get install -y python3-pip git

RUN chown -R 42420:42420 /workspace && \
	addgroup --gid 42420 ovh && \
	useradd --uid 42420 -g ovh --shell /bin/bash -d /workspace ovh

USER ovh

ENV PATH /workspace/.local/bin:$PATH

ADD --chown=ovh:ovh ./requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

ADD --chown=ovh:ovh . /workspace

# run server
CMD python3 app.py --port 8080 --model_version mini
