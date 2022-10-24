FROM apache/beam_python3.7_sdk:2.32.0

COPY . .

RUN python setup.py install

RUN pip install -r requirements-inference.txt

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]