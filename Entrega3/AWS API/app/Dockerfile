FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# RUN pip3 uninstall -r requirements.txt -y

COPY . ${LAMBDA_TASK_ROOT}

CMD ["src.app.lambda_handler"]