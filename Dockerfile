FROM eidos-service.di.unito.it/eidos-base-pytorch:1.11.0

# Copy source files and make it owned by the group eidoslab
# and give write permission to the group
COPY src /src

COPY requirements.txt /src/requirements.txt

RUN chmod 775 /src
RUN chown -R :1337 /src

# Do the same with the data folder
RUN mkdir /data
RUN chmod 775 /data
RUN chown -R :1337 /data

RUN mkdir /scratch
RUN chmod 775 /scratch
RUN chown -R :1337 /scratch

WORKDIR /src

RUN pip install --upgrade -r requirements.txt