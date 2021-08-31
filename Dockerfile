FROM python:3 

COPY license_character_classes.npy .
COPY License_character_recognition_weight.h5 .
COPY MobileNets_character_recognition.json .
COPY wpod-net.h5 .
COPY wpod-net.json .
COPY detect.py .
COPY local_utils.py .

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install keras 
RUN pip install -U scikit-learn
RUN pip install tensorflow

CMD ["python","./detect.py"]
