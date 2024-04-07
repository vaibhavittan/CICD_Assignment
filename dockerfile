FROM python:3.9.7
WORKDIR /CICD_ASSIGNMENT
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python train.py
CMD ["python", "test.py"]