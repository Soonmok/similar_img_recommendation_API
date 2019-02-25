FROM gw000/keras:2.1.4-py2-tf-cpu
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]

