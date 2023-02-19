FROM openjdk:8-jre-alpine

ENV APPLICATION_USER ktor
RUN adduser -D -g '' $APPLICATION_USER

RUN mkdir /app
RUN chown -R $APPLICATION_USER /app

USER $APPLICATION_USER

# TODO: fix
COPY ./build/libs/app-0.0.1-all.jar /app/app.jar
WORKDIR /app

# https://ktor.io/quickstart/quickstart/docker.html
CMD [ \
    "java", \
    "-server", \
    "-XX:+UnlockExperimentalVMOptions", \
    "-XX:+UseCGroupMemoryLimitForHeap", \
    "-XX:InitialRAMFraction=2", \
    "-XX:MinRAMFraction=2", \
    "-XX:MaxRAMFraction=2", \
    "-XX:+UseG1GC", \
    "-XX:MaxGCPauseMillis=100", \
    "-XX:+UseStringDeduplication", \
    "-jar", \
    "app.jar" ]
