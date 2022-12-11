rm -r model/src
cp -r ../model/src model/src
docker build --tag fastapi:latest .
docker run -it --gpus all -p 51819:51819 fastapi:latest
