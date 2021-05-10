# docker build -t taglibapp .
# docker run -p 18080:8080 taglibapp

FROM tomcat
ADD taglibApp /opt/taglibApp
ADD taglibApp/target/taglibApp.war /usr/local/tomcat/webapps/taglibApp.war