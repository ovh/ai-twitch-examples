FROM ovhcom/ai-training-tensorflow:2.4.1

WORKDIR /workspace
ADD . /workspace

RUN pip install -r requirements.txt

CMD [ "streamlit" , "run" , "/workspace/app.py", "--server.address=0.0.0.0" ]

RUN chown -R 42420:42420 /workspace

ENV HOME=/workspace
