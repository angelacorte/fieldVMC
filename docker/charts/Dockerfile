FROM python:3.13.5
RUN mkdir /experiment
WORKDIR /experiment
COPY .python-version ./
COPY process.py ./
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir seaborn matplotlib pandas
CMD python process.py