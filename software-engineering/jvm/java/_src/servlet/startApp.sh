cd testbbs
mvn compile
cd ..
docker build -t servlet .
docker run -ip 18080:8080 servlet