docker login -u hiroga
docker build -t hiroga/terraform-gcp .
# docker run -it --entrypoint /bin/ash hiroga/terraform-gcp
docker push hiroga/terraform-gcp
