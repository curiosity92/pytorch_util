FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# S config timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# E config timezone

# copy all files
WORKDIR /pytorch_util_service
COPY ./requirements.txt /pytorch_util_service/

# install dependencies
RUN pip install -i https://pypi.douban.com/simple/ -r requirements.txt &&\
    rm requirements.txt

# copy all files other than requirements
COPY ./pytorch_util /pytorch_util_service/pytorch_util/
COPY ./server.py /pytorch_util_service/


## S install vim
#RUN apt-get update && apt-get install vim -y
## E install vim


# S expose ports
#EXPOSE 9002
# E expose ports

# S run server
#WORKDIR /dolphin-nlp-service
RUN chmod 777 -R .
#RUN python server.py
# E run server