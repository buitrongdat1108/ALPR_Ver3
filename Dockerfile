FROM python:3 

COPY . .

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install keras 
RUN pip install -U scikit-learn
RUN pip install tensorflow

CMD ["python","./detect.py"]
