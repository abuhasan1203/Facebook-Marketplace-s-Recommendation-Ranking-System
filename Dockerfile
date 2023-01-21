FROM python:3.9

WORKDIR /fb-ml

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:/fb-ml"

EXPOSE 8080

LABEL maintainer="Abu Hasan" \
      version="1.0"

CMD ["python", "app/api.py"]