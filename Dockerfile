FROM python:2.7
ADD data/*.csv /data/
ADD modules/*.py /modules/
ADD titanic.py /
RUN pip install argparse pandas pytest sklearn
CMD [ "python2.7", "-u", "./titanic.py" ]
