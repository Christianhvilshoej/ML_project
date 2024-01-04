FROM python:3.9-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY ML_project/ ML_project/
COPY data/ data/


WORKDIR /
RUN pip install . --no-cache-dir #(1)




ENTRYPOINT ["python","-u","ML_project/models/train.py" ]