gragradle build -p hello
docker build -t struts-hello .
docker run -ip 18080:8080 struts-hello