FROM python:3.11-slim

RUN pip install --no-cache-dir trackio

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

WORKDIR /app
VOLUME ["/root/.cache/huggingface/trackio"]

CMD ["trackio", "show", "--host", "0.0.0.0"]
