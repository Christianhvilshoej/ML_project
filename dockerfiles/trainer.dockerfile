# Base image
FROM python:3.9-slim

#install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#Copy important files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY ML_project/ ML_project/
COPY data/ data/


#Set workdir
WORKDIR /
RUN pip install . --no-cache-dir #(1)

#Make entrypoint
ENTRYPOINT ["python","-u","ML_project/models/train.py" ]