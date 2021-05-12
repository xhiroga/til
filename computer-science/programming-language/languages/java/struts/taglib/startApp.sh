mvn -f taglibApp/pom.xml clean install
docker build -t struts-hello .
docker run -ip 18080:8080 struts-hello