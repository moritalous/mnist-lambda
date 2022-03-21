FROM python:3.9 as model_builder

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py /train.py

WORKDIR /work

RUN python3 /train.py

FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./

RUN python3.9 -m pip install -r requirements.txt -t .

COPY --from=model_builder /work/mnist_model.h5 ./
COPY app.py ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
