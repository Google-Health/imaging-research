FROM apache/beam_python3.7_sdk:2.35.0

COPY . .

RUN pip install cxr-foundation

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]