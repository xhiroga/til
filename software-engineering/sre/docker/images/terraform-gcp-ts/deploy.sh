docker login -u hiroga
docker build -t hiroga/terraform-gcp-ts .
# docker run -it --entrypoint /bin/ash hiroga/terraform-gcp-ts
docker push hiroga/terraform-gcp-ts
